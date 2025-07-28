import bpy
import json


class MMDFacialMorphGroup(bpy.types.Operator):
    bl_idname = "object.mmd_facial_morph_group"
    bl_label = bpy.app.translations.pgettext("Apply Facial Group From JSON")
    bl_description = bpy.app.translations.pgettext("Apply JSON Facial Group to selected mesh.")
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    convert_to_vertex_morph: bpy.props.BoolProperty(
        name=bpy.app.translations.pgettext("Convert to Vertex Morph"),
        description=bpy.app.translations.pgettext("Convert group morphs to vertex morphs after creation."),
        default=False,
    )

    def execute(self, context):
        obj = context.object

        # 找到 MESH 对象
        if obj.type == 'ARMATURE':
            mesh = next((c for c in obj.children if c.type == 'MESH'), None)
        elif obj.type == 'MESH':
            mesh = obj
        else:
            mesh = None

        if not mesh or not hasattr(mesh, "mmd_root") or not hasattr(mesh.mmd_root, "group_morphs"):
            self.report({'ERROR'}, bpy.app.translations.pgettext("当前对象未绑定 mmd_root 或未初始化 Morph，请先在 Morph Tools 中点击 Bind。"))
            return {'CANCELLED'}

        # 收集所有形态键名称
        morph_name_map = {}
        for morph in obj.data.shape_keys.key_blocks:
            key = morph.name.split(".", 1)[-1].lower()
            morph_name_map[key] = morph.name

        # 读取 JSON 文件
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                facial_data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, bpy.app.translations.pgettext("无法读取 JSON 文件: "), e)
            return {'CANCELLED'}

        added_count = 0

        for group_name, parts in facial_data.items():
            if not isinstance(parts, dict):
                continue
            if not any(k in parts for k in ("m", "e", "eye")):
                continue

            morph_list = []

            for key, prefix in [("m", "face_m"), ("e", "face_e"), ("eye", "eye")]:
                if key in parts and isinstance(parts[key], int):
                    index = parts[key]
                    suffix = f"{prefix}{index:02d}"
                    morph_key = suffix.lower()
                    if morph_key in morph_name_map:
                        morph_list.append(morph_name_map[morph_key])

            if not morph_list:
                continue

            # 找到当前顶层对象名称
            obj = bpy.context.active_object
            top_parent = obj
            while top_parent.parent:
                top_parent = top_parent.parent
            active_obj_name = top_parent.name

            active_obj = bpy.data.objects[active_obj_name].mmd_root
            active_obj.active_morph_type = 'group_morphs'

            # 检查是否已有同名 group morph
            group = None
            for g in active_obj.group_morphs:
                if g.name == group_name:
                    group = g
                    g.data.clear()
                    break

            if group is None:
                bpy.ops.mmd_tools.morph_add()
                group = active_obj.group_morphs[active_obj.active_morph]
                group.name = group_name
                group.name_en = group_name
                group.type = 'vertex'

            for morph_name in morph_list:
                item = group.data.add()
                item.name = morph_name
                item.type = 'vertex'
                item.factor = 1.0
                added_count += 1

            if self.convert_to_vertex_morph:
                bpy.ops.mmd_tools.convert_group_morph_to_vertex_morph()

        self.report({'INFO'}, bpy.app.translations.pgettext("✅ 已添加 {count} 项 Morph 到 Group Morph 中").format(count=added_count))
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "convert_to_vertex_morph")


def menu_func(self, context):
    self.layout.operator(MMDFacialMorphGroup.bl_idname, icon='SHAPEKEY_DATA')
