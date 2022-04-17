from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView, CreateView, UpdateView, DeleteView

from .forms import ReviewForm
from .models import Perfume, Preference


class PerfumeListView(ListView):
    model = Perfume
    template_name = 'predictor/perfume_list.html'
    context_object_name = 'perfumes'
    ordering = ['-added_date']
    paginate_by = 10


class UserPerfumeListView(ListView):
    model = Perfume
    template_name = 'predictor/user_perfumes_list.html'
    context_object_name = 'perfumes'
    paginate_by = 10

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Perfume.objects.filter(added_by=user).order_by('-added_date')


def perfume_detail(request, pk):
    perfume = get_object_or_404(Perfume, pk=pk)
    form = ReviewForm()

    return render(request, 'predictor/perfume_detail.html', {'perfume': perfume, 'form': form})


class PerfumeCreateView(LoginRequiredMixin, CreateView):
    model = Perfume
    fields = ['name', 'house', 'description']

    def form_valid(self, form):
        form.instance.added_by = self.request.user
        # setting the author before the validation is run
        return super().form_valid(form)


class PerfumeUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Perfume
    fields = ['name', 'house', 'description']

    def form_valid(self, form):
        form.instance.added_by = self.request.user
        # setting the author before the validation is run
        return super().form_valid(form)

    def test_func(self):
        perfume = self.get_object()
        if self.request.user == perfume.added_by:
            return True
        return False


class PerfumeDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Perfume
    success_url = '/'

    def test_func(self):
        perfume = self.get_object()
        if self.request.user == perfume.added_by:
            return True
        return False


class UserPreferenceListView(ListView):
    model = Preference
    template_name = 'predictor/user_preference_list.html'
    context_object_name = 'preferences'
    paginate_by = 5

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Preference.objects.filter(user=user).order_by('-review_date')

    def test_func(self):
        preference = self.get_object()
        if self.request.user == preference.user:
            return True
        return False








