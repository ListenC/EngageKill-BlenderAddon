import bpy
from bpy.types import Panel, Operator

from . import EngageKillPreset
from . import addon_updater_ops


# 工具栏面板
class EngageKillToolsPanel(Panel):
    bl_label = bpy.app.translations.pgettext("Engage Kill Tools")
    bl_idname = "blendshape_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Engage Kill Tools"

    def draw(self, context):
        layout = self.layout
        layout.label(text=bpy.app.translations.pgettext("Apply BlendShapes From JSON"))
        layout.operator("object.apply_blendshapes", icon='SHAPEKEY_DATA', text=bpy.app.translations.pgettext("Import"))
        layout.label(text=bpy.app.translations.pgettext("Auto Categorize Morphs"))
        layout.operator("object.mmd_categorize_morph", icon='TOOL_SETTINGS', text=bpy.app.translations.pgettext("Automatically assign"))
        layout.label(text=bpy.app.translations.pgettext("Apply Facial Group From JSON"))
        layout.operator("object.mmd_facial_morph_group", icon='SHAPEKEY_DATA', text=bpy.app.translations.pgettext("Import"))
        EngageKillPreset.draw_panel(layout, context)
        # the updater ships a demo panel of its own that used to carry
        # these two calls; it is not registered here, and this panel is
        # the one that gets opened, so the update check lives here now
        addon_updater_ops.check_for_update_background()
        addon_updater_ops.update_notice_box_ui(self, context)