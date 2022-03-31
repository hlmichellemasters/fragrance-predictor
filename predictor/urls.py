from django.urls import path
from .views import (
    PerfumeListView,
    PerfumeDetailView,
    PerfumeCreateView,
    PerfumeUpdateView,
    PerfumeDeleteView,
    UserPerfumeListView)
from . import views


urlpatterns = [
    path('', PerfumeListView.as_view(), name='predictor-home'),
    path('user/<str:username>', UserPerfumeListView.as_view(), name='user-perfumes'),
    path('perfume/<int:pk>/', PerfumeDetailView.as_view(), name='perfume-detail'),
    path('perfume/new/', PerfumeCreateView.as_view(), name='perfume-create'),
    path('perfume/<int:pk>/update/', PerfumeUpdateView.as_view(), name='perfume-update'),
    path('perfume/<int:pk>/delete/', PerfumeDeleteView.as_view(), name='perfume-delete'),
    path('about/', views.about, name='predictor-about'),
]

