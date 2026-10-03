# SPDX-License-Identifier: GPL-3.0-or-later
"""Engage Kill render preset.

Applies the game's unlit NPR shader ``Character/MatcapOutline`` to an imported
PMX model as a Blender node graph.  The character is drawn by that shader with
**no NdotL term at all**, so MMD's ``albedo * diffuse * toon(NdotL) + ambient +
sphere`` pipeline can never reproduce it - every PMX attempt comes out dark
because MMD always folds in a light direction the game never had.  Here the
shader is rebuilt one-to-one out of nodes and the material is *emissive*, so the
render is exactly the shader's output.

The shader, as decompiled (fragment 166-251 of ``Character/MatcapOutline``)::

    m1, m2   = _Matcap1Tex.r, _Matcap2Tex.r     uv = n_view.xy * 0.5 + 0.5
    mh       = _MatcapHighlightTex.r            uv = TEXCOORD2
    Lc       = lerp(max(m1, mh), max(m2, mh), mask.r)
    s        = step(0.5, Lc)
    MulA     = lerp(_Matcap1MulColor.a, _Matcap2MulColor.a, mask.r)
    AddA     = lerp(_Matcap1AddColor.a, _Matcap2AddColor.a, mask.r)
    shade    = 1 + 4*MulA * (2*Lc*(1-s) + s - 1)
    shade    = 1 + vcol.r * (shade - 1)               _ApplyVertexR > 0.5
    cap      = 1 + _MaskGMulCoef*(2*mask.g - 1) + step(0.5, mask.g)
    shade    = max(mask.b, min(shade, cap))
    Mul      = lerp(_Matcap1MulColor.rgb, _Matcap2MulColor.rgb, mask.r)
    base     = clamp(lerp(_MainTex.rgb, _VertexBColor.rgb,
                          (1-vcol.b) * _VertexBColor.a))
    rgb      = base * shade + (1 - shade) * Mul
    light    = step(0.475, mask.g) * 4*AddA * (Lc - 0.5) * vcol.g
    glow     = max((mask.g - 0.5) * _MaskGAddCoef,
                   max(light, mask.b * _MaskBAddColor.a))
    Add      = lerp(lerp(_Matcap1AddColor.rgb, _Matcap2AddColor.rgb, mask.r),
                    _MaskBAddColor.rgb, mask.b)
    rgb     += Add * glow
    rgb     *= lerp(_AmbientLightColor.rgb, 1, mask.b * _AmbientLightColor.w)
    rgb      = clamp(rgb) * (1 - mask.b) + mask.b * rgb

Outline (fragment 12477)::

    colour = texture(_OutlineLUT, uv0).rgb * _AmbientLightColor.rgb

with the hull pushed ``0.0005 * |z_view|`` in screen space, i.e. a constant
~1px rim.  Rebuilt as an inverted hull: a Solidify shell with flipped normals on
the same object, assigned to its own back-face-culled material.

Colour space
------------
The game runs a non-linear (gamma) project: no sRGB decode when sampling, no
re-encode when writing, so the shader's numbers *are* the pixels.  Every texture
is therefore Non-Color and the view transform must be Raw, which writes the
shader output straight into the image.

Where the numbers come from
---------------------------
The node graph is identical for every character, so ``presets/engagekill.json``
is one general preset and holds nothing character-specific:

* the **texture filenames** are derived, not listed.  mmd_tools has already
  loaded the material's main texture out of the PMX, and ``<stem>_mask`` /
  ``<stem>_color_out`` / ``<stem>_high`` sit beside it - so a character or weapon
  prefix follows along by itself, and skirt reusing body's texture also reuses
  body's mask and LUT.
* the **constants** come from the material's *role*, picked by matching the
  texture stem against the preset's ``role_rules`` (hair / skirt / face / wep,
  otherwise body).
* the **matcap** likewise, one name per role: ``Slim2`` for cloth and skin,
  ``Slim1`` for hair and weapons.

A ``presets/characters/<character>.json`` dump wins outright for whichever
materials it names, for a character whose asset differs too much from the
general preset.

Notes for the PMX route
-----------------------
* the albedo, the mask and the outline LUT all read the mesh's **base uv**, and
  its layer name depends on the importer: mmd_tools calls a PMX's ``UVMap`` where
  an FBX import calls it ``UV0``.  It is therefore read off the meshes rather
  than assumed - a ``UV Map`` node naming a layer that does not exist reads texel
  (0, 0), which flattens the entire model to one colour instead of erroring.
* A PMX has no vertex colour field at all, so ``VColRaw`` would come out as
  Unity's default white and the shader's ``shade``, ``_VertexBColor`` and sheen
  terms would all read 1.  The bundle's own colour is dumped per mesh beside the
  hair channel (``vcol_channels_<model>.npz``, shipped in ``cache/`` too) and
  matched on by vertex position in ``rebuild_vcol``, which parks it in a
  ``VColSrc`` byte-colour layer that ``bake_vcol`` then freezes.  A mesh the
  bundle leaves colourless keeps the default white, as before.
* The hair highlight reads ``TEXCOORD2``.  A PMX that still carries that channel
  parks it in ``addUV[0]``, which mmd_tools brings in as ``UV1`` through the same
  ``flipUV_V`` it uses for the base uv - so ``UV1`` already holds the game's
  values and only the vertex shader's clamp has to be re-applied.  That channel is
  trusted only when it is a coherent uv; a distributed model shows what a bad one
  looks like, and it is not rare, because an FBX out of AssetStudio keeps UV0
  alone so a converter that *did* want to write ``addUV`` had nothing to write and
  a converter that wrote something filled it from the wrong reading of TEXCOORD2.
  When the channel is jumbled or absent the sheen uv is rebuilt from the bundle's
  own cache - ``hair_channels_<model>.npz``, located by vertex position
  (``bake_highlight_uv``).  A copy of that cache ships inside the addon, under
  ``cache/``, because the PMX being repaired is usually nowhere near this project.
  A hair that was *reshaped* by hand keeps its vertices in the order the dump
  lists them while moving them, so the position match runs dry and the order is
  read instead (``order_highlight_uv``); only when that is out of reach as well
  does the sheen read ``HIGHLIGHT_MISSING`` (0, *not* Unity's default white, which
  would push ``Lc`` to 1 and kill the whole matcap/Mul machinery).
* A material with no main texture at all (``MatcapFaceNoOutline``, which only
  exists as a runtime-assigned face overlay) has no role and is skipped, as is
  anything whose mask or matcap is not on disk (``face_ef_*``).
"""
import json
import os
import re

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, StringProperty

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# ---- shader constants that are not in the material asset ---------------------
UNITY_COLORSPACE = "gamma"   # "linear" for a project that decodes sRGB on sampling
HIGHLIGHT_MISSING = 0.0      # see the docstring: 0, not Unity's default white
#: a real uv only jumps between faces where the mesh carries a seam.  Measured
#: p90 of the per-face jump (``max|d|`` of the *unclamped* channel): 0.11 for the
#: bundle's own highlight uv, against 2.23 for a PMX whose addUV was filled with
#: TEXCOORD2 read two components at a time - see `bake_highlight_uv`.
HIGHLIGHT_JUMBLE = 0.5
UV_HIGHLIGHT = "UVHigh"      # holds TEXCOORD2 as (u from .w, v from .y)
HIGHLIGHT_U_COMPONENT = 3    # the sheen's u is TEXCOORD2.w
HIGHLIGHT_V_COMPONENT = 1    # ...its v is TEXCOORD2.y, which the shader clamps
DYNAMIC_RANGE = 1.0          # _Chr_DynamicRangeScale
# `_AmbientLightColor` is overwritten per scene through a MaterialPropertyBlock,
# so the character's absolute brightness follows the scene rather than the asset
# (0.902).  Measured against the user's in-game capture it is 1.15x the asset's
# value - one factor that fits flat cloth, matcap gradient and skin alike.
AMBIENT_RGB = (1.04, 1.04, 1.04)
ALBEDO_CS = "sRGB" if UNITY_COLORSPACE == "linear" else "Non-Color"

#: mmd_tools' first additional-uv layer: PMX ``addUV[0]``, i.e. TEXCOORD2, with
#: v flipped (importer.py 649-670).
UV_ADD_XY = "UV1"


def log(*a):
    print("[EKGK]", *a)


def tr(text):
    """Look `text` up in the addon's own string table (``translation.py``).

    That table is keyed with context ``"*"``, which is what ``pgettext`` falls
    back to, so a bare call is the whole lookup.  Only strings the user can read
    go through here - the ``log`` diagnostics stay as they are.
    """
    return bpy.app.translations.pgettext(text)


# ------------------------------------------------------------------ graph
# Built with module-level helpers over the tree being worked on, because
# threading it through every call only adds noise.
TREE = None


def nn(kind, loc):
    nd = TREE.nodes.new(kind)
    nd.location = loc
    return nd


def lnk(sock, target):
    TREE.links.new(sock, target)
    return target


def feed(nd, i, val, vec=False):
    """Wire `val` into input `i`: socket -> link, number -> scalar, tuple -> vector."""
    if val is None:
        return
    if isinstance(val, (int, float)):
        nd.inputs[i].default_value = (float(val),) * 3 if vec else float(val)
    elif isinstance(val, (tuple, list)):
        nd.inputs[i].default_value = (tuple(val) + (0.0, 0.0, 0.0))[:3]
    else:
        lnk(val, nd.inputs[i])


def m(op, a, b, loc, c=None):
    """Scalar math.  Never feed it a colour - use `v` for those."""
    nd = nn("ShaderNodeMath", loc)
    nd.operation = op
    feed(nd, 0, a)
    feed(nd, 1, b)
    feed(nd, 2, c)
    return nd.outputs[0]


def v(op, a, b, loc, c=None):
    """Per-channel math; numbers broadcast to (n,n,n)."""
    nd = nn("ShaderNodeVectorMath", loc)
    nd.operation = op
    feed(nd, 0, a, vec=True)
    feed(nd, 1, b, vec=True)
    feed(nd, 2, c, vec=True)
    return nd.outputs[0]


def mixc(a, b, fac, loc):
    """RGB lerp; the factor is whatever scalar socket."""
    nd = nn("ShaderNodeMix", loc)
    nd.data_type = "RGBA"
    nd.blend_type = "MIX"
    feed(nd, 0, fac)
    feed(nd, 6, a)
    feed(nd, 7, b)
    return nd.outputs[2]


def rgb(value, loc):
    nd = nn("ShaderNodeRGB", loc)
    nd.outputs[0].default_value = (value[0], value[1], value[2], 1.0)
    return nd.outputs[0]


def const(value, loc):
    nd = nn("ShaderNodeValue", loc)
    nd.outputs[0].default_value = float(value)
    return nd.outputs[0]


def clamp01(sock, loc):
    return v("MAXIMUM", v("MINIMUM", sock, (1.0, 1.0, 1.0), loc), (0.0, 0.0, 0.0),
             (loc[0], loc[1] - 80))


# ------------------------------------------------------------------ inputs
class Shader:
    """The per-material inputs the node graph needs, read from the preset."""

    def __init__(self, params, tex_root, warn, has_highlight=True, uv_name=""):
        self.p = params
        self.tex_root = tex_root
        self.warn = warn
        self.images = {}
        #: the layer the albedo/mask/LUT are addressed with, read off the mesh -
        #: see `main_uv_name`
        self.uv_name = uv_name
        #: no TEXCOORD2 in the mesh, so the sheen must read a constant 0 rather
        #: than the texture: Unity's default white would put Lc at 1 everywhere.
        self.has_highlight = has_highlight

    def image(self, key, colorspace):
        name = self.p["tex"].get(key)
        if not name:
            return None
        if key == "_MatcapHighlightTex" and not self.has_highlight:
            return None
        cached = self.images.get(name)
        if cached is None:
            path = os.path.join(self.tex_root, name + ".png")
            if not os.path.exists(path):
                self.warn.append("missing texture %s" % path)
                return None
            cached = bpy.data.images.load(path, check_existing=True)
            cached.colorspace_settings.name = colorspace
            self.images[name] = cached
        return cached

    def col(self, key, default=(0.0, 0.0, 0.0, 0.0)):
        return tuple(self.p["colors"].get(key, list(default)))

    def flt(self, key, default=0.0):
        return float(self.p["floats"].get(key, default))

    def ambient(self):
        """`_AmbientLightColor`, with the scene override in `AMBIENT_RGB`."""
        amb = list(self.col("_AmbientLightColor", (1.0, 1.0, 1.0, 0.0)))
        if AMBIENT_RGB is not None:
            amb[:3] = AMBIENT_RGB
        return amb


def tex_node(image, vector, loc, uvname=None, extension="EXTEND"):
    nd = nn("ShaderNodeTexImage", loc)
    nd.image = image
    nd.interpolation = "Linear"
    nd.extension = extension
    if vector is None:
        if uvname is None:
            raise TypeError("tex_node: name the uv layer the sampler reads")
        uv = nn("ShaderNodeUVMap", (loc[0] - 220, loc[1]))
        uv.uv_map = uvname
        lnk(uv.outputs[0], nd.inputs[0])
    else:
        lnk(vector, nd.inputs[0])
    return nd


# ------------------------------------------------------------------ shader
def build_shader(mat, sh):
    """Rewrite `mat` as the game's unlit MatcapOutline shader."""
    global TREE
    mat.use_nodes = True
    TREE = mat.node_tree
    TREE.nodes.clear()

    x = -1500
    out = nn("ShaderNodeOutputMaterial", (4400, -80))
    emit = nn("ShaderNodeEmission", (4200, -80))
    lnk(emit.outputs[0], out.inputs[0])

    # --- uv0, and the matcap uv = view-space normal projected to the unit disc
    uvmap = nn("ShaderNodeUVMap", (x, 300))
    uvmap.uv_map = sh.uv_name
    geo = nn("ShaderNodeNewGeometry", (x, 620))
    vtx = nn("ShaderNodeVectorTransform", (x + 240, 620))
    vtx.vector_type = "NORMAL"
    vtx.convert_from = "WORLD"
    vtx.convert_to = "CAMERA"
    lnk(geo.outputs["Normal"], vtx.inputs[0])
    matuv = v("MULTIPLY_ADD", vtx.outputs[0], (0.5, 0.5, 0.5), (x + 480, 620),
              c=(0.5, 0.5, 0.5))

    # --- vertex colour, raw (see bake_vcol)
    vcol = nn("ShaderNodeAttribute", (x, 120))
    vcol.attribute_type = "GEOMETRY"
    vcol.attribute_name = "VColRaw"

    main = sh.image("_MainTex", ALBEDO_CS)
    mask = sh.image("_MaskTex", "Non-Color")
    cap1 = sh.image("_Matcap1Tex", "Non-Color")
    cap2 = sh.image("_Matcap2Tex", "Non-Color")
    high = sh.image("_MatcapHighlightTex", "Non-Color")

    mainn = tex_node(main, uvmap.outputs[0], (x, -140)) if main else None
    maskn = tex_node(mask, uvmap.outputs[0], (x, -420)) if mask else None
    cap1n = tex_node(cap1, matuv, (x + 480, 430)) if cap1 else None
    cap2n = tex_node(cap2, matuv, (x + 480, 430)) if cap2 else None
    # TEXCOORD2, off its own uv map - not the matcap projection.  This is the one
    # Repeat sampler on the whole character: the u axis tiles there (which is what
    # makes `_MatcapHighlightOffsetU`'s sweep seamless), and clamping instead
    # folds half the streak onto the texture's first and last columns.
    highn = (tex_node(high, None, (x + 480, 760), uvname=UV_HIGHLIGHT,
                      extension="REPEAT")
             if high else None)

    white = const(1.0, (x + 700, 460))

    def sep(node, which):
        if node is None:
            return white
        nd = nn("ShaderNodeSeparateColor", (node.location[0] + 240,
                                           node.location[1] + 40))
        nd.mode = "RGB"
        lnk(node.outputs["Color"], nd.inputs[0])
        return nd.outputs[which]

    main_rgb = mainn.outputs["Color"] if mainn else const(1.0, (x + 700, -140))
    mr = sep(maskn, "Red")
    mg = sep(maskn, "Green")
    mb = sep(maskn, "Blue")
    cr1 = sep(cap1n, "Red")
    cr2 = sep(cap2n, "Red")
    crh = (sep(highn, "Red") if highn
           else const(HIGHLIGHT_MISSING, (x + 700, 560)))

    vcs = nn("ShaderNodeSeparateColor", (x + 240, 120))
    vcs.mode = "RGB"
    lnk(vcol.outputs["Color"], vcs.inputs[0])
    vcr, vcg, vcb = (vcs.outputs[c] for c in ("Red", "Green", "Blue"))

    # --- material constants
    m1mul, m2mul = sh.col("_Matcap1MulColor"), sh.col("_Matcap2MulColor")
    m1add, m2add = sh.col("_Matcap1AddColor"), sh.col("_Matcap2AddColor")
    mbadd = sh.col("_MaskBAddColor")
    vbcol = sh.col("_VertexBColor")
    amb = sh.ambient()
    keep_vcol_r = sh.flt("_ApplyVertexR") > 0.5
    gmul, gadd = sh.flt("_MaskGMulCoef", 1.0), sh.flt("_MaskGAddCoef", 1.0)

    # --- Lc = lerp(max(m1, mh), max(m2, mh), mask.r)
    lc1 = m("MAXIMUM", cr1, crh, (x + 960, 620))
    lc2 = m("MAXIMUM", cr2, crh, (x + 960, 520))
    lc = m("MULTIPLY_ADD", mr, m("SUBTRACT", lc2, lc1, (x + 1160, 560)),
           (x + 1340, 580), c=lc1)
    s = m("GREATER_THAN", lc, 0.5, (x + 1520, 520))
    mula = m("MULTIPLY_ADD", mr, const(m2mul[3] - m1mul[3], (x + 1160, 420)),
             (x + 1340, 400), c=const(m1mul[3], (x + 1160, 360)))
    adda = m("MULTIPLY_ADD", mr, const(m2add[3] - m1add[3], (x + 1160, 260)),
             (x + 1340, 240), c=const(m1add[3], (x + 1160, 200)))

    # --- shade = 1 + 4*MulA*(2*Lc*(1-s) + s - 1), then the vertex-red lerp
    inner = m("MULTIPLY_ADD", m("MULTIPLY", lc, 2.0, (x + 1520, 700)),
              m("SUBTRACT", 1.0, s, (x + 1520, 660)), (x + 1700, 680), c=s)
    shade = m("MULTIPLY_ADD", m("MULTIPLY", mula, 4.0, (x + 1520, 400)),
              m("SUBTRACT", inner, 1.0, (x + 1520, 620)), (x + 1880, 520), c=1.0)
    if keep_vcol_r:
        shade = m("MULTIPLY_ADD", vcr,
                  m("SUBTRACT", shade, 1.0, (x + 2060, 460)), (x + 2220, 460),
                  c=1.0)

    # --- cap = 1 + _MaskGMulCoef*(2*mask.g - 1) + step(0.5, mask.g)
    cap = m("ADD", m("MULTIPLY_ADD", mg, 2.0 * gmul, (x + 1520, 160),
                     c=1.0 - gmul),
            m("GREATER_THAN", mg, 0.5, (x + 1520, 100)), (x + 1700, 140))
    shade = m("MAXIMUM", mb, m("MINIMUM", shade, cap, (x + 2060, 300)),
              (x + 2220, 300))
    oms = m("SUBTRACT", 1.0, shade, (x + 2400, 300))

    # --- base = clamp(lerp(_MainTex, _VertexBColor, (1-vcol.b)*_VertexBColor.a))
    bfac = m("MULTIPLY", m("SUBTRACT", 1.0, vcb, (x + 960, -240)), vbcol[3],
             (x + 1160, -240))
    base = mixc(main_rgb, rgb(vbcol[:3], (x + 960, -400)), bfac, (x + 1340, -300))
    base = clamp01(base, (x + 1520, -300))

    # --- rgb = base * shade_final + (1 - shade_final) * Mul
    mul = mixc(rgb(m1mul[:3], (x + 960, 60)), rgb(m2mul[:3], (x + 960, -20)),
               mr, (x + 1180, 20))
    col = v("MULTIPLY_ADD", mul, oms, (x + 2800, -60),
            c=v("MULTIPLY", base, shade, (x + 2600, -60)))

    # --- glow = max((mask.g-0.5)*_MaskGAddCoef,
    #                max(step(0.475, mask.g)*4*AddA*(Lc-0.5)*vcol.g,
    #                    mask.b*_MaskBAddColor.a))
    light = m("MULTIPLY",
              m("MULTIPLY", adda, 4.0, (x + 960, 900)),
              m("MULTIPLY", m("GREATER_THAN", mg, 0.475, (x + 960, 840)),
                m("MULTIPLY", m("SUBTRACT", lc, 0.5, (x + 1340, 800)), vcg,
                  (x + 1520, 820)), (x + 1700, 860)), (x + 1880, 880))
    glow = m("MAXIMUM",
             m("MULTIPLY", m("SUBTRACT", mg, 0.5, (x + 1160, 940)), gadd,
               (x + 1340, 920)),
             m("MAXIMUM", light, m("MULTIPLY", mb, mbadd[3], (x + 1880, 1020)),
               (x + 2060, 1000)), (x + 2220, 960))
    add = mixc(mixc(rgb(m1add[:3], (x + 960, -560)),
                    rgb(m2add[:3], (x + 960, -640)), mr, (x + 1180, -600)),
               rgb(mbadd[:3], (x + 960, -760)), mb, (x + 1400, -660))
    col = v("ADD", col, v("MULTIPLY", add, glow, (x + 3000, -220)),
            (x + 3200, -160))

    # --- rgb *= lerp(_AmbientLightColor.rgb, 1, mask.b*_AmbientLightColor.w)
    ambw = m("MULTIPLY", mb, amb[3], (x + 3000, 140))
    ambmix = mixc(rgb(amb[:3], (x + 3000, 20)),
                  rgb((1.0, 1.0, 1.0), (x + 3000, -60)), ambw, (x + 3300, 0))
    col = v("MULTIPLY", col, ambmix, (x + 3500, -140))

    # --- rgb = clamp(rgb)*(1-mask.b) + mask.b*rgb
    final = mixc(clamp01(col, (x + 3800, -240)), col, mb, (x + 3900, -100))
    if DYNAMIC_RANGE != 1.0:
        final = v("MULTIPLY", final, (DYNAMIC_RANGE,) * 3, (x + 4060, -100))
    lnk(final, emit.inputs[0])

    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return TREE


def build_outline_shader(mat, sh):
    """`texture(_OutlineLUT, uv0).rgb * _AmbientLightColor.rgb` (fragment 12477)."""
    global TREE
    mat.use_nodes = True
    TREE = mat.node_tree
    TREE.nodes.clear()
    lut = sh.image("_OutlineLUT", ALBEDO_CS)
    if lut is None:
        return None
    out = nn("ShaderNodeOutputMaterial", (600, 0))
    emit = nn("ShaderNodeEmission", (400, 0))
    lnk(emit.outputs[0], out.inputs[0])
    tex = tex_node(lut, None, (-900, 0), uvname=sh.uv_name)
    amb = sh.ambient()
    lnk(v("MULTIPLY", tex.outputs["Color"], amb[:3], (-600, 0)),
        emit.inputs[0])
    mat.use_backface_culling = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return TREE


# ------------------------------------------------------------------ outline
def drop_outline(obj):
    """Remove a hull from an earlier run so re-applying stays idempotent."""
    for mod in [m for m in obj.modifiers if m.name == "outline"]:
        obj.modifiers.remove(mod)
    for mat in [m for m in obj.data.materials if m and m.name.startswith(obj.name + "_outline")]:
        obj.data.materials.pop(index=list(obj.data.materials).index(mat))
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def outline_depth_ratio(obj):
    """Distance to the object's bounds centre over distance to its origin.

    ``blender_char.set_outline`` measures its rim against the *model centre* -
    the framing distance - while a driver's ``LOC_DIFF`` can only see the object
    origin.  The two differ by however much of the model sits off the origin, so
    the ratio is measured once here and baked into the expression as a constant;
    it survives a camera move because both distances grow together while the
    shot keeps the model in frame.  Returns 1.0 when the camera is at the origin
    or the bounds are degenerate.
    """
    cam = bpy.context.scene.camera
    if cam is None:
        return 1.0
    mw = np.array(obj.matrix_world)
    local = np.array([tuple(c) + (1.0,) for c in obj.bound_box])
    world = (local @ mw.T)[:, :3]
    eye = np.array(cam.matrix_world.translation)
    origin = np.array(obj.matrix_world.translation)
    span = float(np.linalg.norm(origin - eye))
    if span < 1e-6:
        return 1.0
    return float(np.linalg.norm(world.mean(axis=0) - eye)) / span


def drive_outline(mod, obj, px):
    """Hold the rim at `px` pixels of the rendered image, whatever the framing.

    Solidify offsets by a world-space distance, so one fixed number cannot be
    right for two framings at once - which is why an object-space value comes
    out as a hairline in a full-body shot and a thick band in a close-up.  The
    game's hull is ``0.0005 * |z_view|`` in screen space and therefore always
    the same number of pixels, and ``blender_char.set_outline`` reproduces that
    with ``width = px * dist * sensor / (lens * resolution_x)``, solved from
    ``ndc = x*lens/(z*sensor/2)``.  The same expression drives the modifier
    here, so it follows the camera instead of being baked for one of them.

    ``dist`` is the camera-to-origin distance; `outline_depth_ratio` folds in
    the bit of the model that sits off the origin, so the rim is measured where
    the pipeline measures it.  The focal terms are read through the scene,
    so lens, resolution and even the camera datablock can change freely; the
    camera *object* is bound by hand, so switching to another one needs the
    preset re-applied.
    """
    scene = bpy.context.scene
    drv = mod.driver_add("thickness").driver
    drv.type = "SCRIPTED"

    dist = drv.variables.new()
    dist.name = "dist"
    dist.type = "LOC_DIFF"
    dist.targets[0].id = obj
    dist.targets[1].id = scene.camera

    def prop(name, path):
        var = drv.variables.new()
        var.name = name
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "SCENE"
        var.targets[0].id = scene
        var.targets[0].data_path = path

    prop("lens", "camera.data.lens")
    prop("sensor", "camera.data.sensor_width")
    prop("rx", "render.resolution_x")
    prop("ry", "render.resolution_y")
    # AUTO sensor fit puts `sensor_width` on the longer axis, so the slice of it
    # spanning x shrinks in a portrait render - the `min` of set_outline
    drv.expression = ("%g * %g * dist * sensor * min(1, rx / ry) / (lens * rx)"
                      % (px, outline_depth_ratio(obj)))
    return drv


def add_outline(obj, sh, px):
    """Inverted hull as a Solidify shell on the same object, so that it follows
    the shape keys.  The shell's normals are flipped and its material is
    back-face culled, which leaves only the outer rim past the depth test.

    Returns None when the scene has no camera: `px` is only meaningful against
    one, and the driver would evaluate to zero without it."""
    if bpy.context.scene.camera is None:
        return None
    out_mat = bpy.data.materials.new(obj.name + "_outline")
    if build_outline_shader(out_mat, sh) is None:
        bpy.data.materials.remove(out_mat)
        return None
    obj.data.materials.append(out_mat)
    mod = obj.modifiers.new("outline", "SOLIDIFY")
    drive_outline(mod, obj, px)
    mod.offset = 1.0
    mod.use_rim = False
    mod.use_even_offset = False
    if hasattr(mod, "use_flip_normals"):
        mod.use_flip_normals = True
    # every face has to land on the last slot, whatever its original index
    mod.material_offset = len(obj.data.materials)
    mod.material_offset_rim = len(obj.data.materials)
    return mod


# ------------------------------------------------------------------ geometry
def bake_vcol(obj):
    """Freeze the raw vertex colour into a FLOAT_COLOR attribute named VColRaw.

    Unity hands mesh colours to the shader as raw Color32 bytes / 255 with no
    sRGB decode, and the shader uses them as plain numbers.  Blender's colour
    attributes are colour managed, so the raw values are copied into a new
    FLOAT_COLOR attribute (`color_srgb` is exactly bytes/255) and the graph reads
    that instead.  A PMX has no vertex colour field to copy from, so unless
    `rebuild_vcol` has already put the bundle's own one in place (as a `VColSrc`
    byte colour) the attribute is created filled with Unity's default white.
    """
    me = obj.data
    src = me.color_attributes.active_color if me.color_attributes else None
    count = len(me.loops) if (src is not None and len(src.data) != len(me.vertices)) \
        else len(me.vertices)
    raw = [1.0] * (count * 4)
    if src is not None:
        src.data.foreach_get("color_srgb", raw)
    dst = me.color_attributes.get("VColRaw") or me.color_attributes.new(
        name="VColRaw", type="FLOAT_COLOR",
        domain=src.domain if src else "POINT")
    dst.data.foreach_set("color", raw)
    return src is not None


#: `src/uv3_extract.py` dumps mesh_hair's position, uv0 and TEXCOORD2 here
HAIR_CACHE = "hair_channels_%s.npz"
#: ...and the addon keeps a copy of those dumps for the characters it knows, in a
#: folder beside itself, so the sheen can be rebuilt for a PMX that lives anywhere
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
#: the same command dumps every mesh's vertex colour here as well, because a PMX
#: has no field to keep one - `rebuild_vcol` matches it on by position
VCOL_CACHE = "vcol_channels_%s.npz"
#: the layer `rebuild_vcol` parks the recovered colour in, for `bake_vcol`
VCOL_SRC = "VColSrc"

#: a hair vertex lands on the bundle's own to 1e-7 m under `bundle = (-x, z, -y)`,
#: while the soup's vertices sit ~5 mm apart, so a millimetre cannot reach a
#: neighbour yet still tolerates a small edit
MATCH_TOL = 1e-3
#: ...and uv0 tells the soup's coincident vertices apart
MATCH_UV0_TOL = 1e-4
#: share of the dump a scene has to hold before it is the same hair mesh
MATCH_COVER = 0.9
#: per-vertex pre-filter; 5 mm keeps the fine pass down to the hair's own loops
MATCH_NEAR = 5e-3
#: share of the dump that has to sit on its own *index* to a millimetre before a
#: hair the position match could not reach is read off the order instead.  A
#: reshaped hair keeps its untouched strands exactly where the dump lists them
#: (chr_033_003's reshaped build: 908 of 3082, the median strand 14 mm off), and
#: no other mesh lands a quarter of its vertices on another mesh's, in order, to
#: a millimetre by accident.
ORDER_COVER = 0.25

STEM_RE = re.compile(r"(chr_\d+_\d+_\d+)")


def model_stem(obj):
    """The character name a ``hair_channels_<stem>.npz`` dump is keyed on.

    Read off the object's own material textures, which is where the preset takes
    the model's filenames from anyway; a weapon material sharing the mesh names
    ``wep_...``, so a ``chr_...`` hit wins.
    """
    for slot in obj.material_slots:
        main = main_texture_name(slot.material) if slot.material else None
        hit = STEM_RE.match(main or "")
        if hit:
            return hit.group(1)
    hit = STEM_RE.match(obj.name)
    return hit.group(1) if hit else None


def cache_roots(context):
    """Folders that may hold a ``hair_channels_<stem>.npz``.

    Three places are worth a look.  The addon ships its own copy under ``cache/``
    and that is the one that always works: a distributed PMX usually sits nowhere
    near this project, so a lookup that only knows about the scene would come back
    empty and leave the sheen at 0 - which reads as "the preset did nothing to the
    hair".  The scene's own folder and its texture folder, each with their
    parents, are searched as well, so a dump that has just been regenerated for a
    model the addon does not carry is still found.
    """
    roots = []
    for base in (os.path.dirname(bpy.data.filepath),
                 bpy.path.abspath(context.scene.ek_tex_dir or ""),
                 CACHE_DIR):
        if not base:
            continue
        folder = os.path.abspath(base)
        for _ in range(4):
            if folder not in roots:
                roots.append(folder)
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
    return roots


def find_cache(name, context):
    """A dump called `name`, out of the folders `cache_roots` looks at.

    The same three places the sheen uv is looked for, and for the same reason:
    the addon's own ``cache/`` is the copy that always works, while the scene's
    folder and its texture folder cover a dump just regenerated by hand.
    """
    for root in cache_roots(context):
        for rel in ("", "output_blender"):
            path = os.path.join(root, rel, name)
            if os.path.exists(path):
                return path
    return None


def find_channels(stem, context):
    """The cached TEXCOORD2 dump for `stem`, or None."""
    return find_cache(HAIR_CACHE % stem, context) if stem else None


def pmx_highlight_uv(obj):
    """The sheen uv as this PMX carries it, or None if it carries no usable one.

    mmd_tools parks a PMX's ``addUV[0]`` in ``UV1`` and pushes it through the
    same ``flipUV_V`` as the base uv (importer.py 653), so when that channel *is*
    the sheen's uv the layer already reads in the game's convention: u as-is, v
    as-is, and only the shader's clamp is left to re-apply.

    A uv only jumps between faces where the mesh has a seam, so a channel whose
    faces are jagged is not a uv at all.  The test runs before the clamp, where the
    jumbled channel is at its most obviously wrong - clipping v to 0..1 folds part
    of that jump away and would flatter it - and it counts only faces that carry
    the channel, because a PMX keeps every other vertex at "no extra uv", which
    `flipUV_V` turns into the pair (0, 1) rather than (0, 0).  p90 rather than the
    mean, so a handful of seam faces cannot blame a good channel: the bundle's own
    uv scores 0.11 where the shipped chr_033 PMX scores 2.23.

    Returns ``(per-loop uv, note)``, or None when the layer is absent, empty or
    jumbled - the caller then rebuilds it from the bundle.
    """
    me = obj.data
    xy = me.uv_layers.get(UV_ADD_XY)
    if xy is None or len(xy.data) != len(me.loops):
        return None
    n = len(me.loops)
    b = np.empty(n * 2, dtype=np.float32)
    xy.data.foreach_get("uv", b)
    if not b.any():
        return None
    uv = np.empty((n, 2), dtype=np.float32)
    uv[:, 0] = b[0::2]                            # TEXCOORD2.u, tiles -> Repeat
    uv[:, 1] = b[1::2]                            # ...and v, still unclamped
    carries = np.abs(uv - (0.0, 1.0)).max(axis=1) > 1e-6
    jumps = [np.abs(np.diff(uv[p.loop_start:p.loop_start + p.loop_total], axis=0)).max()
             for p in me.polygons
             if carries[p.loop_start:p.loop_start + p.loop_total].any()]
    jump = float(np.percentile(jumps, 90)) if jumps else 0.0
    if jump > HIGHLIGHT_JUMBLE:
        log("  uvhigh %-18s rejected, not a uv (%.2f jump per face)"
            % (obj.name, jump))
        return None
    return uv, "addUV (%.2f jump per face)" % jump


def rebuild_highlight_uv(obj, stem, context, base_uv):
    """Put the real sheen uv back on the mesh, out of the bundle's cached channel.

    A PMX can only carry ``addUV`` when the scene it was exported from had an
    extra uv layer - and one imported from AssetStudio's FBX never does, that
    export keeps UV0 alone - so whatever a PMX does carry came from elsewhere.  A
    distributed model shows what that looks like: TEXCOORD2 consumed two
    components at a time, one vertex holding some other vertex's (.x, .y) and the
    next its (.z, .w), v unclamped at -1.94..1.50 where the sheen's own channel
    spans -0.24..2.94 only *after* the shader clamps it, and the sheen band's
    ``step(0.5, Lc)`` boundary landing on every triangle - white shards over the
    whole head.  Rejected, the sheen reads 0 and the hair stays flat pink.

    The numbers are in the bundle, so they are looked up by *where a vertex is*
    rather than by index: ``src/uv3_extract.py`` dumps mesh_hair's position, uv0
    and TEXCOORD2 per model, and a vertex is matched on position (agreeing to
    1e-7 m) with uv0 breaking ties between the soup's coincident vertices.  A
    scene that cannot be matched within a millimetre, or that holds less than
    nine tenths of the dump's *distinct* vertices - the repeats a hard edge
    forces are collapsed first, the way `rebuild_vcol` does it - is not this
    character's hair and gets nothing.

    Returns ``(per-loop uv, note)``, or None.
    """
    path = find_channels(stem, context)
    if path is None:
        log("  uvhigh %-18s no %s in reach, sheen stays 0"
            % (obj.name, HAIR_CACHE % (stem or "<model>")))
        return None
    try:
        z = np.load(path)
        src = np.asarray(z["pos"], np.float64)
        suv0 = np.asarray(z["uv0"], np.float64)
        uv2 = np.asarray(z["uv2"], np.float64)
    except Exception as error:
        log("  uvhigh %-18s %s unusable: %s" % (obj.name, path, error))
        return None

    # a bundle mesh repeats a vertex once per hard edge, so one (position,
    # uv0) turns up several times - chr_032's hair holds 4800 entries but
    # only 3581 distinct ones - while a PMX vertex carries a single sheen uv.
    # Collapsing the repeats onto their first entry is what makes the
    # coverage test below count against the vertices the dump really has:
    # judged against the raw 4800 a faithful match reads as 75% and is thrown
    # away, which is what silenced chr_032's sheen.  `order_highlight_uv`
    # still takes the raw dump, because the order it reads is written in it.
    keep = np.unique(np.hstack([src, suv0]), axis=0, return_index=True)[1]
    msrc, msuv0, muv2 = src[keep], suv0[keep], uv2[keep]

    me = obj.data
    n = len(me.vertices)
    loops = len(me.loops)
    if n == 0 or loops == 0:
        return None
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    m = np.asarray(obj.matrix_world, dtype=np.float64)
    world = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    # mmd_tools maps MMD's left-handed y-up onto Blender's z-up, so the bundle's
    # own axes come back out as (-x, z, -y) - measured in src/_p5.py
    here = np.stack([-world[:, 0], world[:, 2], -world[:, 1]], 1)

    layer = me.uv_layers.get(base_uv)
    if layer is None:
        log("  uvhigh %-18s no %s layer to match the dump with"
            % (obj.name, base_uv))
        return None
    luv = np.empty(loops * 2, dtype=np.float32)
    layer.data.foreach_get("uv", luv)
    luv = luv.reshape(-1, 2).astype(np.float64)
    vert = np.empty(loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", vert)

    # coarse pass: which vertices are on the hair at all (the rest of the mesh is
    # a different material and has nothing to do), then the exact pass per loop
    near = np.full(n, np.inf)
    for s in range(0, n, 1024):
        e = min(n, s + 1024)
        d = np.linalg.norm(here[s:e, None, :] - msrc[None, :, :], axis=2)
        near[s:e] = d.min(1)
    cand = np.flatnonzero((near < MATCH_NEAR)[vert])
    take = np.full(loops, -1, dtype=np.int32)
    for s in range(0, len(cand), 256):
        rows = cand[s:s + 256]
        d = np.linalg.norm(here[vert[rows], None, :] - msrc[None, :, :], axis=2)
        du = np.abs(luv[rows, None, :] - msuv0[None, :, :]).sum(2)
        du[d >= MATCH_TOL] = np.inf
        k = du.argmin(1)
        hit = du[np.arange(len(rows)), k] < MATCH_UV0_TOL
        take[rows[hit]] = k[hit]

    found = np.unique(take[take >= 0])
    if len(found) < MATCH_COVER * len(msrc):
        # the positions do not carry this mesh, but the *order* may: a hair that
        # was reshaped in place keeps its vertices where the dump lists them
        got = order_highlight_uv(obj, here, src, uv2)
        if got is not None:
            return got
        log("  uvhigh %-18s %s is not this hair (%d/%d verts by position), "
            "sheen stays 0" % (obj.name, path, len(found), len(msrc)))
        return None
    uv = np.zeros((loops, 2), dtype=np.float32)
    ok = take >= 0
    uv[ok, 0] = muv2[take[ok], HIGHLIGHT_U_COMPONENT]
    # clamp exactly as the vertex shader does (line 79:
    # `vs_TEXCOORD3.y = clamp(vs_TEXCOORD3.y, 0.0, 1.0)`).  Left unclamped, v
    # swings to -0.24..2.94 and the single bright streak in the sheen texture
    # repeats all over the head instead of staying in its band.
    uv[ok, 1] = np.clip(muv2[take[ok], HIGHLIGHT_V_COMPONENT], 0.0, 1.0)
    return uv, "rebuilt from %s (%d/%d verts)" % (path, len(found), len(msrc))


def hair_vertices(obj):
    """The vertices of the object's hair material, in the mesh's own order.

    ``hair_channels_<model>.npz`` is the bundle's ``mesh_hair``, so the block
    of vertices it corresponds to is the one the hair material's faces name,
    and its order is the PMX's vertex order - which is the order the dump was
    written in, because an export of ours writes the bundle's hair mesh out as
    one run.

    Returns the mesh vertex indices as an int array, or None when the object
    carries no plain hair material (or more than one, which would leave two
    blocks claiming the same order).
    """
    me = obj.data
    slots = []
    for i, slot in enumerate(obj.material_slots):
        main = main_texture_name(slot.material) if slot.material else None
        if not main:
            continue
        stem = main[:-6] if main.endswith("_color") else main
        if stem.endswith("_hair"):
            slots.append(i)
    if len(slots) != 1:
        return None
    used = set()
    for poly in me.polygons:
        if poly.material_index == slots[0]:
            used.update(poly.vertices)
    return np.array(sorted(used), dtype=np.int64) if used else None


def order_highlight_uv(obj, here, src, uv2):
    """The sheen uv read off the dump's vertex *order*, for a reshaped hair.

    The position match is what makes the rebuild work on a mesh an exporter put
    back together, but it needs the hair to still be *there*.  A model reshaped
    by hand - chr_033_003's is 14 mm off at the median strand, its longest 110
    mm - keeps only its untouched strands in place, so the match covers a third
    of the dump and is thrown away, and the sheen then reads 0 while the hair
    stays flat.  The vertex order survives that edit and the dump is written in
    it, so the k-th hair vertex takes the k-th entry.

    Believing the order unchecked would put another mesh's sheen onto this one,
    so it has to show: ``ORDER_COVER`` of the dump sitting on its own index to a
    millimetre.  A reshaped hair clears that on the strands it did not move, by
    a wide margin (29% against the 25% asked); a mesh that merely shares the
    vertex count cannot, because its vertices would have to coincide.

    Returns ``(per-loop uv, note)``, or None.
    """
    used = hair_vertices(obj)
    if used is None or len(used) != len(src):
        log("  uvhigh %-18s no hair block to read the order off (%s verts vs "
            "%d in the dump)"
            % (obj.name, "none" if used is None else len(used), len(src)))
        return None
    inplace = int((np.linalg.norm(here[used] - src, axis=1) < MATCH_TOL).sum())
    if inplace < ORDER_COVER * len(src):
        log("  uvhigh %-18s order is not this hair (%d/%d verts on their own "
            "index)" % (obj.name, inplace, len(src)))
        return None
    me = obj.data
    vert = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vert)
    k = np.searchsorted(used, vert)
    hit = (k < len(used)) & (used[np.minimum(k, len(used) - 1)] == vert)
    uv = np.zeros((len(me.loops), 2), dtype=np.float32)
    uv[hit, 0] = uv2[k[hit], HIGHLIGHT_U_COMPONENT]
    uv[hit, 1] = np.clip(uv2[k[hit], HIGHLIGHT_V_COMPONENT], 0.0, 1.0)
    return uv, "order of %d/%d verts on their own index" % (inplace, len(src))


def bake_highlight_uv(obj, stem=None, context=None, base_uv=None):
    """Give the mesh a ``UVHigh`` layer holding ``TEXCOORD2``, or say it cannot.

    The hair sheen is the one thing about this shader the asset does not carry
    through a normal import: it reads ``TEXCOORD2``, whose u tiles (-0.5..1.5
    here) and whose v runs well past 1 until the vertex shader clamps it, while
    the PMX format has no such channel and the FBX export throws it away.  So the
    channel is taken from wherever the scene has it - the PMX's own ``addUV``
    when that holds a coherent uv, and otherwise the bundle's dump, matched by
    position (`rebuild_highlight_uv`).

    Returns False when the sheen legitimately has nothing to read, so the
    material can be told to treat the sheen as 0 instead of sampling a constant.
    """
    me = obj.data
    got = pmx_highlight_uv(obj)
    if got is None:
        got = rebuild_highlight_uv(obj, stem, context, base_uv)
    if got is None:
        return False
    uv, note = got
    layer = me.uv_layers.get(UV_HIGHLIGHT) or me.uv_layers.new(name=UV_HIGHLIGHT)
    layer.name = UV_HIGHLIGHT
    layer.data.foreach_set("uv", uv.ravel())
    log("  uvhigh %-18s %s  u %.2f..%.2f  v %.2f..%.2f"
        % (obj.name, note, uv[:, 0].min(), uv[:, 0].max(),
           uv[:, 1].min(), uv[:, 1].max()))
    return True


def rebuild_vcol(obj, stem, context, base_uv):
    """Put the bundle's own vertex colour back on the mesh, as ``VColSrc``.

    A PMX vertex is a position, a normal, two uvs, up to four extra uvs and the
    skin weights, and nothing else: there is no colour field anywhere in the
    format.  So a model imported from one comes in white however faithful the
    export was, while the shader leans on ``vcol`` in three places (``shade``,
    the ``_VertexBColor`` blend and the sheen's ``step``) - a ``face_e`` ends up
    twice as bright, a ``face_m`` five times, and ``hair_trans`` loses the pink
    it is tinted with.  That is the whole of why the PMX and the FBX the preset
    was written against disagree.

    ``src/uv3_extract.py`` dumps every mesh of the bundle whose colour is not
    plain white, and the values are matched onto the PMX the same way the sheen
    uv is: by *where a vertex is*, to a millimetre under ``bundle = (-x, z, -y)``,
    with uv0 breaking the tie between vertices that sit on top of each other.  A
    bundle mesh repeats a vertex once per hard edge and a PMX vertex holds one
    colour, so those repeats are collapsed onto their first entry before the
    match.  A mesh the bundle leaves colourless has no entry, so its vertices
    find nothing and stay white - which is what the game draws them as.

    The attribute is BYTE_COLOR on purpose.  ``bake_vcol`` reads ``color_srgb``,
    which for a byte colour is the stored byte over 255 - the raw numbers Unity
    hands the shader - while a float colour would take a linear-to-sRGB curve on
    the way out and shift every value.  Eight bits is also all the source has: the
    bundle's colour is a Color32 to begin with, so the round trip is exact.

    Returns True when the mesh carries the dump, so the caller knows the colour
    is the model's own; False leaves it white, exactly as before this existed.
    """
    path = find_cache(VCOL_CACHE % stem, context) if stem else None
    if path is None:
        log("  vcol  %-18s no %s in reach, colour stays white"
            % (obj.name, VCOL_CACHE % (stem or "<model>")))
        return False
    try:
        z = np.load(path)
        src = np.asarray(z["pos"], np.float64)
        suv0 = np.asarray(z["uv0"], np.float64)
        col = np.asarray(z["col"], np.uint8)
    except Exception as error:
        log("  vcol  %-18s %s unusable: %s" % (obj.name, path, error))
        return False
    if len(src) == 0:
        log("  vcol  %-18s %s is empty - this model has no colour of its own"
            % (obj.name, path))
        return False

    # a bundle mesh repeats a vertex once per hard edge, so one (position, uv0)
    # turns up several times - 630 of chr_033's 6082 entries - while a PMX
    # vertex holds a single colour.  Collapsing the repeats onto their first
    # entry also makes the coverage test below count against the distinct
    # vertices the dump really has (5452, not 6082); without this a faithful
    # match reads as 89% and is discarded.
    key = np.hstack([src, suv0])
    keep = np.unique(key, axis=0, return_index=True)[1]
    src, suv0, col = src[keep], suv0[keep], col[keep]

    me = obj.data
    n = len(me.vertices)
    loops = len(me.loops)
    if n == 0 or loops == 0:
        return False
    layer = me.uv_layers.get(base_uv)
    if layer is None:
        log("  vcol  %-18s no %s layer to match the dump with"
            % (obj.name, base_uv))
        return False

    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    m = np.asarray(obj.matrix_world, dtype=np.float64)
    world = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    # mmd_tools maps MMD's left-handed y-up onto Blender's z-up, so the bundle's
    # own axes come back out as (-x, z, -y) - measured in src/_p5.py
    here = np.stack([-world[:, 0], world[:, 2], -world[:, 1]], 1)

    luv = np.empty(loops * 2, dtype=np.float32)
    layer.data.foreach_get("uv", luv)
    luv = luv.reshape(-1, 2).astype(np.float64)
    vert = np.empty(loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", vert)
    # a vertex holds one colour however many loops land on it, so a single uv0
    # per vertex breaks the tie with; walking the loops backwards leaves the
    # smallest loop index of each vertex in `first`
    first = np.zeros(n, dtype=np.int32)
    first[vert[::-1]] = np.arange(loops, dtype=np.int32)[::-1]

    # coarse pass: which vertices are on a coloured mesh at all (the rest of the
    # model is a different mesh and carries no colour), then the exact pass
    near = np.full(n, np.inf)
    for s in range(0, n, 1024):
        e = min(n, s + 1024)
        d = np.linalg.norm(here[s:e, None, :] - src[None, :, :], axis=2)
        near[s:e] = d.min(1)
    cand = np.flatnonzero(near < MATCH_NEAR)
    take = np.full(n, -1, dtype=np.int32)
    for s in range(0, len(cand), 256):
        rows = cand[s:s + 256]
        d = np.linalg.norm(here[rows, None, :] - src[None, :, :], axis=2)
        du = np.abs(luv[first[rows], None, :] - suv0[None, :, :]).sum(2)
        du[d >= MATCH_TOL] = np.inf
        k = du.argmin(1)
        hit = du[np.arange(len(rows)), k] < MATCH_UV0_TOL
        take[rows[hit]] = k[hit]

    found = np.unique(take[take >= 0])
    if len(found) < MATCH_COVER * len(src):
        log("  vcol  %-18s %s is not this model (%d/%d verts matched), colour "
            "stays white" % (obj.name, path, len(found), len(src)))
        return False

    raw = np.ones((n, 4), dtype=np.float32)
    ok = take >= 0
    raw[ok] = col[take[ok]] / 255.0
    dst = me.color_attributes.get(VCOL_SRC) or me.color_attributes.new(
        name=VCOL_SRC, type="BYTE_COLOR", domain="POINT")
    dst.data.foreach_set("color_srgb", raw.ravel())
    me.color_attributes.active_color = dst
    log("  vcol  %-18s rebuilt from %s (%d/%d verts)  %s"
        % (obj.name, path, len(found), len(src),
           " ".join("rgb"[i] + "[%.2f,%.2f]"
                    % (raw[:, i].min(), raw[:, i].max()) for i in range(3))))
    return True


# ------------------------------------------------------------------ presets
CHARACTER_DIR = os.path.join(PRESET_DIR, "characters")

#: what a material cannot be built without; the outline LUT is optional
REQUIRED_TEX = ("_MainTex", "_MaskTex", "_Matcap1Tex")


def preset_items(self, context):
    if not os.path.isdir(PRESET_DIR):
        return [("", "(no presets)", "")]
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR)
                   if f.lower().endswith(".json"))
    if not names:
        return [("", "(no presets)", "")]
    return [(n, n, "presets/%s.json" % n) for n in names]


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_preset(name):
    return load_json(os.path.join(PRESET_DIR, name + ".json"))


def load_override(meshes):
    """An optional ``presets/characters/<character>.json`` override.

    mmd_tools names the root ``<pmx stem>``, the armature ``<pmx stem>_arm`` and
    the mesh ``<pmx stem>_mesh``, so the PMX's own name is the first
    candidate.  It is not the only one: a file that has been through an
    edit chain carries that chain in its name
    (``chr_032_001_01_fbx_fixcz..._fixcz2.pmx``) while the dump the
    preset was written from is keyed on the character alone, so the name
    read off the materials wins as a fallback.  Without it the override is
    missed and the generic role constants are used instead, which tints the
    model with another character's colours.
    """
    if not os.path.isdir(CHARACTER_DIR):
        return None, None
    for obj in meshes:
        names = [obj.name]
        for suffix in ("_mesh", "_arm"):
            if obj.name.endswith(suffix):
                names.append(obj.name[:-len(suffix)])
        character = model_stem(obj)
        if character:
            names.append(character)
        for stem in names:
            path = os.path.join(CHARACTER_DIR, stem + ".json")
            if os.path.exists(path):
                return stem, load_json(path)
    return None, None


def main_texture_name(mat):
    """The texture mmd_tools loaded for this material, without its extension."""
    if mat.use_nodes and mat.node_tree:
        for nd in mat.node_tree.nodes:
            if nd.type == "TEX_IMAGE" and nd.image:
                return os.path.splitext(os.path.basename(
                    nd.image.filepath or nd.image.name))[0]
    return None


#: the layer a PMX's base uv lives on.  mmd_tools calls it ``UVMap``; only an
#: FBX import names it ``UV0``.  Getters, in preference order.
UV0_CANDIDATES = ("UVMap", "UV0")


def main_uv_name(meshes):
    """The uv layer the albedo, mask and outline LUT are addressed with.

    A ``UV Map`` node pointed at a layer the mesh does not have reads texel
    (0, 0) - one flat colour over the whole model rather than a visible fault.
    The name is therefore read off the meshes instead of assumed: the first
    candidate the meshes actually carry, preferring the one every mesh shares.
    """
    every = [set(l.name for l in o.data.uv_layers) for o in meshes]
    shared = set.intersection(*every) if every else set()
    for name in UV0_CANDIDATES:
        if name in shared:
            return name
    for name in UV0_CANDIDATES:
        for layers in every:
            if name in layers:
                return name
    # nothing expected survived: keep mmd_tools' name if any mesh has a layer,
    # so the log at least names what the graph will ask for
    for name in UV0_CANDIDATES:
        for layers in every:
            if any(l for l in layers):
                return name
    return UV0_CANDIDATES[0]


def role_of(stem, preset):
    for rule in preset["role_rules"]:
        if rule.get("contains") and rule["contains"] in stem:
            return rule["role"]
        if rule.get("prefix") and stem.startswith(rule["prefix"]):
            return rule["role"]
    return preset["fallback_role"]


def resolve_material(mat, preset, override):
    """This material's constants and texture filenames, or None to skip it.

    An override wins outright where it names the material; otherwise everything
    is derived - the filenames from the material's own main texture, so they
    follow whatever character or weapon prefix the PMX happens to use.
    """
    if override:
        props = override.get("materials", {}).get(mat.name)
        if props is not None:
            # an override still has to name a renderable material - one with no
            # main texture (MatcapFaceNoOutline) is a runtime overlay, not a pass
            return props if props["tex"].get("_MainTex") else None
    main = main_texture_name(mat)
    if main is None:
        return None
    suffix = preset["texture"]["main_suffix"]
    stem = main[:-len(suffix)] if main.endswith(suffix) else main
    props = preset["roles"].get(role_of(stem, preset))
    if props is None:
        return None
    naming = preset["texture"]
    tex = {
        "_MainTex": main,
        "_MaskTex": stem + naming["mask_suffix"],
        "_OutlineLUT": stem + naming["outline_suffix"],
        "_Matcap1Tex": props.get("matcap"),
        "_Matcap2Tex": props.get("matcap"),
    }
    # only the roles whose asset actually assigns a sheen map get `<stem>_high`;
    # deriving one for cloth would name a file that never existed
    if props.get("highlight"):
        tex["_MatcapHighlightTex"] = stem + naming["highlight_suffix"]
    return {"colors": props.get("colors", {}), "floats": props.get("floats", {}),
            "tex": tex}


def absent_textures(props, tex_root):
    return [props["tex"][key] for key in REQUIRED_TEX
            if props["tex"].get(key)
            and not os.path.exists(os.path.join(tex_root,
                                                props["tex"][key] + ".png"))]


def has_texture(props, key, tex_root):
    name = props["tex"].get(key)
    return bool(name) and os.path.exists(os.path.join(tex_root, name + ".png"))


#: A material the user faded out by hand - the blush and face-red overlays cut
#: to alpha 0 so an expression can bring them in - is the one that must not get
#: a hull.  The game's *own* transparent materials keep theirs: that variant is
#: a shader which still draws the outline (and still names the same
#: ``_OutlineLUT``), and nothing in the asset tells it apart from the opaque one
#: anyway - `_Alpha`, `_SrcBlend`, `_DstBlend` and `_ZWrite` are the same on
#: ``chr_033_001_01_hair`` and its ``_hair_trans`` sibling, while
#: ``_TransparentFromTex`` is a mode enum that reads 3.0 on plain opaque hair.
#: What is left is the opacity the PMX itself carries, which mmd_tools hands
#: over as the material's alpha and which is 0 on exactly those overlays.
def is_transparent(mat):
    return float(mat.diffuse_color[3]) < 1.0


def outline_source(planned, built, tex_root):
    """The material that colours this object's hull, or why there is none.

    The hull is one per object, so it takes its colour from the first material
    that can really give it one, which means a ``_OutlineLUT`` that is a file in
    the texture folder: `resolve_material` *derives* that name from the main
    texture, so a material whose asset never shipped a ``*_color_out`` would
    otherwise be given a shell coloured by a texture that is not there - and the
    game draws no outline for it either.  A material the user faded out by hand
    is passed over as well: its hull would be a solid rim around geometry that
    is meant to fade out.

    Returns ``(shader, "")`` when a hull can be built, otherwise ``(None, msgid)``
    with the message to report.
    """
    transparent = False
    for mat, props in planned:
        if mat.name not in built:
            continue
        if is_transparent(mat):
            transparent = True
            continue
        if has_texture(props, "_OutlineLUT", tex_root):
            return built[mat.name], ""
    return None, ("No outline on %s: the material is transparent" if transparent
                  else "No outline on %s: no outline texture on disk")


def material_ready(props, folder):
    """Does `folder` hold everything this material cannot be built without?"""
    return all(os.path.exists(os.path.join(folder, props["tex"][key] + ".png"))
               for key in REQUIRED_TEX if props["tex"].get(key))


def resolve_tex_root(context, planned):
    """Find the folder that holds the textures the planned materials need.

    Candidates are the folder the user picked, any folder Blender has already
    loaded one of those textures from (mmd_tools reads them from the PMX's own
    ``textures``), and folders under the blend file.  A candidate wins by how
    many materials it can satisfy *in full* rather than by satisfying all of
    them at once: the materials that end up skipped (the facial overlays
    ``face_ef_*``) derive filenames such as ``<stem>_mask`` that were never
    shipped, and demanding those would reject the one correct folder.
    """
    def score(folder):
        if not folder:
            return 0
        return sum(1 for props in planned if material_ready(props, folder))

    best, best_score, why = None, 0, ""

    def consider(folder, label):
        nonlocal best, best_score, why
        found = score(folder)
        if found > best_score:
            best, best_score, why = folder, found, label

    consider(bpy.path.abspath(context.scene.ek_tex_dir or ""), "面板指定")
    for img in bpy.data.images:
        folder = os.path.dirname(bpy.path.abspath(img.filepath or ""))
        if folder:
            consider(folder, "已加载贴图 %s" % img.name)
    if bpy.data.filepath:
        start = os.path.dirname(bpy.data.filepath)
        for root, dirs, _files in os.walk(start):
            if root.count(os.sep) - start.count(os.sep) > 3:
                dirs[:] = []
                continue
            for d in list(dirs):
                consider(os.path.join(root, d), os.path.join(root, d))
    return best, why


def target_meshes(context):
    sel = [o for o in context.selected_objects if o.type == "MESH" and o.data]
    if sel:
        return sel
    return [o for o in context.scene.objects if o.type == "MESH" and o.data]


# ------------------------------------------------------------------ operator
class EKGK_OT_render_preset(bpy.types.Operator):
    """Apply the Engage Kill character shader to the imported PMX model"""
    bl_idname = "object.ek_render_preset"
    bl_label = "Apply Engage Kill Render Preset"
    bl_description = ("Rebuild every material as the game's unlit "
                      "Character/MatcapOutline shader (matcap + mask + outline)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        name = scene.ek_preset
        if not name:
            self.report({"ERROR"}, tr("No preset available"))
            return {"CANCELLED"}
        try:
            preset = load_preset(name)
        except Exception as error:
            self.report({"ERROR"}, tr("Failed to read the preset: %s") % error)
            return {"CANCELLED"}

        meshes = target_meshes(context)
        if not meshes:
            self.report({"ERROR"},
                        tr("No mesh in the scene, import a PMX model first"))
            return {"CANCELLED"}

        for obj in meshes:
            drop_outline(obj)          # so a re-run does not plan its own hull

        warn, skipped, plan, planned = [], [], {}, []
        character, override = load_override(meshes)
        if character:
            log("角色覆盖 presets/characters/%s.json" % character)
        else:
            # the generic roles are one character's numbers, so a model
            # rendered off them comes out tinted rather than
            # unlit-faulty.  Say so instead of letting it pass as "the
            # preset ran".
            stem = None
            for obj in meshes:
                stem = model_stem(obj)
                if stem:
                    break
            warn.append(tr("no presets/characters/%s.json - the generic role "
                           "constants belong to another character, so the "
                           "colours may not be this model's; "
                           "generate one from the model's own material dump")
                        % (stem or "<model>"))
        for obj in meshes:
            plan[obj] = []
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None:
                    continue
                props = resolve_material(mat, preset, override)
                if props is None:
                    skipped.append(mat.name)
                    continue
                plan[obj].append((mat, props))
                planned.append(props)

        if not planned:
            self.report({"ERROR"},
                        tr("No material can be applied; check that the "
                           "materials carry a main texture"))
            return {"CANCELLED"}
        tex_root, why = resolve_tex_root(context, planned)
        if tex_root is None:
            self.report({"ERROR"},
                        tr("Cannot find the texture folder (the one holding "
                           "%s); set \"Texture Folder\" in the panel")
                        % ", ".join(sorted({p["tex"]["_MainTex"]
                                            for p in planned})[:2]))
            return {"CANCELLED"}
        log("预设 %s  贴图目录 %s (%s)" % (name, tex_root, why))

        uv0 = main_uv_name(meshes)
        log("基础 uv 图层 %s" % uv0)

        done = 0
        for obj in meshes:
            stem = model_stem(obj)
            # the colour has to be in place before bake_vcol freezes it
            rebuild_vcol(obj, stem, context, uv0)
            bake_vcol(obj)
            has_high = bake_highlight_uv(obj, stem, context, uv0)
            built = {}
            for mat, props in plan[obj]:
                lack = absent_textures(props, tex_root)
                if lack:
                    warn.append(tr("%s is missing the texture %s")
                                % (mat.name, ", ".join(lack)))
                    skipped.append(mat.name)
                    continue
                if has_texture(props, "_MatcapHighlightTex", tex_root) \
                        and not has_high:
                    warn.append(tr("%s carries a highlight texture, but this "
                                   "PMX's addUV is not a usable highlight uv, "
                                   "so the sheen is taken as 0") % mat.name)
                sh = built.get(mat.name)
                if sh is None:
                    sh = built[mat.name] = Shader(props, tex_root, warn,
                                                  has_high, uv0)
                build_shader(mat, sh)
                done += 1
            # the hull rides on the material set, one per object
            sh, why = outline_source(plan[obj], built, tex_root)
            if sh is None:
                warn.append(tr(why) % obj.name)
            elif bpy.context.scene.camera is None:
                warn.append(tr("No camera in the scene - outline skipped"))
            elif add_outline(obj, sh, float(scene.ek_outline)) is None:
                warn.append(tr("%s cannot be outlined: its outline texture "
                               "failed to load") % obj.name)

        try:
            scene.view_settings.view_transform = "Raw"
            scene.view_settings.look = "None"
        except Exception as error:
            warn.append(tr("Failed to set the view transform: %s") % error)

        for line in warn:
            log("  !! %s" % line)
        log("材质 %d 个，跳过 %d 个" % (done, len(skipped)))
        self.report({"INFO"},
                    tr("Applied \"%s\": %d materials, %d skipped%s")
                    % (name, done, len(skipped),
                       tr("; %d notes in the console") % len(warn) if warn
                       else ""))
        return {"FINISHED"}


CLASSES = (EKGK_OT_render_preset,)


#: The rim is a fixed number of *pixels*, not a fixed size: the game pushes its
#: hull by ``0.0005 * |z_view|`` in screen space, so a close-up shows the same
#: line as a full-body shot.  Solidify offsets in world space instead, so the
#: control is the pixel count and drive_outline() solves the offset for the
#: camera in use - a baked value is a hairline at one distance and a thick band
#: at another.
DEFAULT_OUTLINE = 1.2

#: the names below are the msgids the panel looks up - the panel passes its own
#: translated ``text``, so these stay in English and the label follows the
#: interface language instead of being frozen when the property is registered.
SCENE_PROPS = (
    ("ek_preset", "Enum", dict(name="Render Preset", items=preset_items)),
    ("ek_tex_dir", "String", dict(name="Texture Folder", subtype="DIR_PATH",
                                  default="")),
    ("ek_outline", "Float", dict(name="Outline Width (px)",
                                 default=DEFAULT_OUTLINE,
                                 min=0.0, max=8.0, precision=2)),
)


def _register_scene_props():
    for name, kind, options in SCENE_PROPS:
        maker = {"Enum": EnumProperty, "String": StringProperty,
                 "Float": FloatProperty}[kind]
        setattr(bpy.types.Scene, name, maker(**options))


def _unregister_scene_props():
    for name, _kind, _options in SCENE_PROPS:
        try:
            delattr(bpy.types.Scene, name)
        except Exception:
            pass


# ------------------------------------------------------------------ ui
def draw_panel(layout, context):
    box = layout.box()
    box.label(text=tr("Engage Kill Render Preset"), icon='SHADERFX')
    box.prop(context.scene, "ek_preset", text="")
    box.operator(EKGK_OT_render_preset.bl_idname,
                 text=tr("One-Click Render Preset"), icon="PLAY")
    row = box.row(align=True)
    row.prop(context.scene, "ek_tex_dir", text=tr("Texture Folder"))
    row.prop(context.scene, "ek_outline", text=tr("Outline Width (px)"))
    if not (context.scene.ek_tex_dir or bpy.data.filepath):
        box.row(align=True).label(text=tr("Texture folder is auto-detected "
                                          "when left empty"), icon="QUESTION")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _register_scene_props()


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_scene_props()