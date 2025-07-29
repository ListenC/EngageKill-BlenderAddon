# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "apply_blendshapes",
    "author": "Bilibili 凡人就行",
    "version": (1, 3, 1),
    "blender": (3, 0, 0),
    "location": "Object > Apply BlendShapes From JSON",
    "description": "Apply Unity-style JSON blendshapes with multi-language support.",
    "category": "Object",
    "support": "COMMUNITY",
    "license": "GPL-3.0",
    "dependencies": ["blender-addon-updater"],
    "addon_updater": {
        "user": "CGCookie",
        "repo": "blender-addon-updater",
        "current_version": (1, 1, 1),
        "use_releases": True,
    }
}

import bpy
from .apply_blendshapes import ApplyBlendshapes
from .MMDCategorizeMorph import MMDCategorizeMorph
from .MMDFacialMorphGroup import MMDFacialMorphGroup
from .EngageKillToolsPanel import EngageKillToolsPanel
from .translation import translation_dict
from .preferences import EngageKillAddonPreferences
from . import addon_updater_ops


def register():
    addon_updater_ops.register(bl_info)
    for cls in EngageKillAddonPreferences:
        addon_updater_ops.make_annotations(cls)  # Avoid blender 2.8 warnings.
        bpy.utils.register_class(cls)
    bpy.utils.register_class(ApplyBlendshapes)
    bpy.utils.register_class(MMDCategorizeMorph)
    bpy.utils.register_class(MMDFacialMorphGroup)
    bpy.utils.register_class(EngageKillToolsPanel)
    bpy.app.translations.register(__name__, translation_dict)
    #bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    addon_updater_ops.unregister()
    for cls in reversed(EngageKillAddonPreferences):
        bpy.utils.unregister_class(cls)
    bpy.utils.unregister_class(ApplyBlendshapes)
    bpy.utils.unregister_class(MMDCategorizeMorph)
    bpy.utils.unregister_class(MMDFacialMorphGroup)
    bpy.utils.unregister_class(EngageKillToolsPanel)
    bpy.app.translations.unregister(__name__)
    #bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()
