# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Apply BlendShapes From JSON",
    "author": "Bilibili 凡人就行",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "Object > Apply BlendShapes From JSON",
    "description": "Apply Unity-style JSON blendshapes with multi-language support.",
    "category": "Object"
}

import bpy
import json
from bpy.props import StringProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector

# 🔑 多语言字典
translation_dict = {
    "en_US": {
        ("*", "Apply BlendShapes From JSON"): "Apply BlendShapes From JSON",
        ("*", "Apply JSON BlendShapes to selected mesh."): "Apply JSON BlendShapes to selected mesh.",
        ("*", "Apply Unity-style JSON blendshapes with multi-language support."): "Apply Unity-style JSON blendshapes with multi-language support.",
        ("*", "Global Scale Factor"): "Global Scale Factor",
        ("*", "Overall scale for vertex positions"): "Overall scale for vertex positions",
    },
    "zh_CN": {
        ("*", "Apply BlendShapes From JSON"): "从JSON应用表情形态",
        ("*", "Apply JSON BlendShapes to selected mesh."): "将JSON表情形态应用到选中网格",
        ("*", "Apply Unity-style JSON blendshapes with multi-language support."): "支持多语言的Unity风格JSON表情形态导入",
        ("*", "Global Scale Factor"): "全局缩放系数",
        ("*", "Overall scale for vertex positions"): "用于顶点位置的整体缩放",
    },
    "zh_TW": {
        ("*", "Apply BlendShapes From JSON"): "從JSON套用表情形態",
        ("*", "Apply JSON BlendShapes to selected mesh."): "將JSON表情形態套用到選取的網格",
        ("*", "Apply Unity-style JSON blendshapes with multi-language support."): "支援多語言的Unity風格JSON表情形態導入",
        ("*", "Global Scale Factor"): "全域縮放係數",
        ("*", "Overall scale for vertex positions"): "用於頂點位置的整體縮放",
    },
    "ja_JP": {
        ("*", "Apply BlendShapes From JSON"): "JSONからブレンドシェイプを適用",
        ("*", "Apply JSON BlendShapes to selected mesh."): "選択したメッシュにJSONのブレンドシェイプを適用します",
        ("*", "Apply Unity-style JSON blendshapes with multi-language support."): "多言語対応のUnityスタイルJSONブレンドシェイプを適用",
        ("*", "Global Scale Factor"): "全体スケール係数",
        ("*", "Overall scale for vertex positions"): "頂点位置の全体スケール",
    },
}


class ApplyBlendshapesOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "object.apply_blendshapes"
    bl_label = bpy.app.translations.pgettext("Apply BlendShapes From JSON")
    bl_description = bpy.app.translations.pgettext("Apply JSON BlendShapes to selected mesh.")

    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    scale_factor: FloatProperty(
        name=bpy.app.translations.pgettext("Global Scale Factor"),
        description=bpy.app.translations.pgettext("Overall scale for vertex positions"),
        default=1.0,
        min=0.0001
    )

    def execute(self, context):
        self.apply_blendshapes(self.filepath, self.scale_factor)
        return {'FINISHED'}

    def apply_blendshapes(self, json_path, scale_factor):
        with open(json_path, 'r') as f:
            data = json.load(f)

        def convert_coordinates(vec_dict, is_normal=False):
            x = vec_dict['x'] * scale_factor
            y = vec_dict['y'] * scale_factor
            z = vec_dict['z'] * scale_factor
            if is_normal:
                return Vector((-x, y, z)).normalized()
            return Vector((-x, y, z))

        for mesh_data in data['data']['meshes']:
            mesh_name = mesh_data['name']
            obj = bpy.data.objects.get(mesh_name)

            if not obj or obj.type != 'MESH':
                self.report({'WARNING'}, f"找不到网格对象: {mesh_name}")
                continue

            mesh = obj.data

            for poly in mesh.polygons:
                poly.use_smooth = True

            if not mesh.shape_keys:
                basis_key = obj.shape_key_add(name="Basis")
                basis_key.interpolation = 'KEY_LINEAR'

            for shape_data in mesh_data['spahes']:
                shape_name = shape_data['name']

                if mesh.shape_keys and shape_name in mesh.shape_keys.key_blocks:
                    shape_key = mesh.shape_keys.key_blocks[shape_name]
                else:
                    shape_key = obj.shape_key_add(name=shape_name)
                    shape_key.interpolation = 'KEY_LINEAR'
                    shape_key.value = 1.0

                delta_vertices = shape_data['deltaVertices']
                if len(delta_vertices) != len(mesh.vertices):
                    self.report({'WARNING'}, f"顶点数不匹配: JSON有{len(delta_vertices)}位移, 网格有{len(mesh.vertices)}位点")
                    continue

                basis_key = mesh.shape_keys.key_blocks["Basis"]

                for i, delta in enumerate(delta_vertices):
                    base_co = basis_key.data[i].co.copy()
                    converted_delta = convert_coordinates(delta)
                    shape_key.data[i].co = base_co + converted_delta

                if shape_name == "Basis" and 'deltaNormals' in shape_data:
                    delta_normals = shape_data['deltaNormals']
                    normals = []
                    for normal in delta_normals:
                        normals.append(convert_coordinates(normal, is_normal=True))

                    mesh.create_normals_split()
                    mesh.normals_split_custom_set_from_vertices(normals)
                    mesh.free_normals_split()
                    mesh.use_auto_smooth = True
                    mesh.auto_smooth_angle = 3.14159

                if mesh.uv_layers:
                    mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)

                shape_key.value = 0.0

        bpy.context.view_layer.update()
        self.report({'INFO'}, "✅ 混合形状处理完成！")


def menu_func(self, context):
    label = bpy.app.translations.pgettext("Apply BlendShapes From JSON")
    self.layout.operator(ApplyBlendshapesOperator.bl_idname, text=label)


def register():
    bpy.utils.register_class(ApplyBlendshapesOperator)
    bpy.app.translations.register(__name__, translation_dict)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.utils.unregister_class(ApplyBlendshapesOperator)
    bpy.app.translations.unregister(__name__)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()
