import bpy
import re


class MMDCategorizeMorph(bpy.types.Operator):
    bl_idname = "object.mmd_categorize_morph"
    bl_label = bpy.app.translations.pgettext("Auto Categorize Morphs")
    bl_description = bpy.app.translations.pgettext("Automatically assign MMD morph type (mouth/eye/etc)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # 获取选中的PMX模型
        obj = context.object
        if not obj or obj.type != 'MESH' or not hasattr(obj, "mmd_root"):
            self.report({'ERROR'}, bpy.app.translations.pgettext("Please select the PMX model"))
            return {'CANCELLED'}

        # 确保有形态键数据
        if not obj.data.shape_keys:
            self.report({'ERROR'}, bpy.app.translations.pgettext("Models have no BlendShapes"))
            return {'CANCELLED'}

        count = 0

        # 找到当前顶层对象名称
        obj = bpy.context.active_object
        top_parent = obj
        while top_parent.parent:
            top_parent = top_parent.parent
        active_obj_name = top_parent.name
        active_obj = bpy.data.objects[active_obj_name].mmd_root

        # 处理顶点形态键
        for morph in active_obj.vertex_morphs:
            # 提取点号后面的部分作为表情名称
            name_parts = morph.name.split('.')
            if len(name_parts) > 1:
                morph_name = name_parts[-1].lower()
            else:
                morph_name = morph.name.lower()

            # 嘴部表情 (口) - 类别 'MOUTH'
            if morph_name.startswith('face_m'):
                morph.category = 'MOUTH'
                count += 1

            # 眼部表情 (目) - 类别 'EYE'
            elif morph_name.startswith('face_e') or morph_name.startswith('eye'):
                morph.category = 'EYE'
                count += 1

        self.report({'INFO'}, bpy.app.translations.pgettext("Completed: {count} emoticons have been classified").format(count=count))
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(MMDCategorizeMorph.bl_idname, icon='SHAPEKEY_DATA')
