from django.contrib import admin
from .models import Profile
from import_export.admin import ImportExportMixin


@admin.register(Profile)
class ProfileAdmin(ImportExportMixin, admin.ModelAdmin):
    model = Profile
