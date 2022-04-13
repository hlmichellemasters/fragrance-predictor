from django.contrib import admin
from .models import Perfume, Preference


class PreferenceAdmin(admin.ModelAdmin):
    model = Preference
    list_display = ('user', 'perfume', 'love', 'comment', 'review_date')
    list_filter = ['user', 'perfume', 'love']
    search_fields = ['comment']


class PerfumeAdmin(admin.ModelAdmin):
    model = Perfume
    list_display = ('name', 'house', 'description', 'added_date', 'added_by')
    list_filter = ['added_by', 'house']
    search_fields = ['name', 'description']


# Register your models here.
admin.site.register(Perfume, PerfumeAdmin)
admin.site.register(Preference, PreferenceAdmin)

