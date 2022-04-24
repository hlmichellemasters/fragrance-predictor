from django.contrib import admin
from .models import Perfume, House, Preference
from import_export.admin import ImportExportMixin


@admin.register(Preference)
class PreferenceAdmin(ImportExportMixin, admin.ModelAdmin):
    model = Preference
    list_display = ('id', 'user', 'perfume', 'love', 'comment', 'review_date')
    list_filter = ('user', 'perfume', 'love')
    search_fields = ('id', 'comment')


@admin.register(Perfume)
class PerfumeAdmin(ImportExportMixin, admin.ModelAdmin):
    model = Perfume
    list_display = ('id', 'name', 'house', 'description', 'added_date', 'added_by')
    list_filter = ('added_by', 'house')
    search_fields = ('id', 'name', 'description')


@admin.register(House)
class HouseAdmin(ImportExportMixin, admin.ModelAdmin):
    model = House
    list_display = ('id', 'name')
    search_fields = ('name',)
