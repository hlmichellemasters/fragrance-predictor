from django.urls import path
from .views import (
    PerfumeListView,
    #PerfumeDetailView,
    PerfumeCreateView,
    PerfumeUpdateView,
    PerfumeDeleteView,
    UserPerfumeListView,
    UserPreferenceListView,
)
from . import views


urlpatterns = [
    path('', PerfumeListView.as_view(), name='predictor-perfumes'),
    # path('perfume/<int:pk>/', PerfumeDetailView.as_view(), name='perfume-detail'),
    path('perfume/<int:pk>/', views.perfume_detail, name='perfume-detail'),

    path('perfume/new/', PerfumeCreateView.as_view(), name='perfume-create'),
    # path('perfume/new/<str:pk>/', PerfumeCreateView.as_view(), name='perfume-create'), (stopped working with pk)
    path('perfume/<int:pk>/update/', PerfumeUpdateView.as_view(), name='perfume-update'),
    path('perfume/<int:pk>/delete/', PerfumeDeleteView.as_view(), name='perfume-delete'),

    path('user/perfume/<str:username>/', UserPerfumeListView.as_view(), name='user-perfumes'),

    # view for past entered preferences of perfumes
    path('user/preferences/<str:username>/', UserPreferenceListView.as_view(), name='user-preference-list'),

    path('perfume/<int:pk>/add_review/', views.add_preference, name='add-review'),

    # provides the view for recommended perfumes
    path('recommendation/', views.user_recommendation_list, name='user-recommendation-list'),

    # path('perfume/<int:pk>/', views.preference_detail(), name='preference_detail'),

    path('about/', views.about, name='predictor-about'),
]

