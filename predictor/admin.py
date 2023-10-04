from django.contrib import admin
from .models import Perfume, House, Preference
from import_export.admin import ImportExportModelAdmin


@admin.register(Preference)
class PreferenceAdmin(ImportExportModelAdmin):
    from_encoding = 'utf8'
    to_encoding = "utf8"
    model = Preference
    list_display = ('id', 'user', 'perfume', 'love', 'comment', 'review_date')
    list_filter = ('user', 'perfume', 'love')
    search_fields = ('id', 'comment')


@admin.register(Perfume)
class PerfumeAdmin(ImportExportModelAdmin):
    from_encoding = 'utf8'
    to_encoding = "utf8"
    model = Perfume
    list_display = ('id', 'name', 'house', 'description',
                    'added_date', 'added_by')
    list_filter = ('added_by', 'house')
    search_fields = ('id', 'name', 'description')


@admin.register(House)
class HouseAdmin(ImportExportModelAdmin):
    from_encoding = 'utf8'
    to_encoding = "utf8"
    model = House
    list_display = ('id', 'name')
    search_fields = ('name',)
