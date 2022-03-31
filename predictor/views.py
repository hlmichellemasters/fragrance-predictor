from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .models import Perfume


def home(request):
    context = {
        'perfumes': Perfume.objects.all()
    }

    return render(request, 'predictor/home.html', context)


class PerfumeListView(ListView):
    model = Perfume
    template_name = 'predictor/home.html'
    context_object_name = 'perfumes'
    ordering = ['-added_date']
    paginate_by = 5


class UserPerfumeListView(ListView):
    model = Perfume
    template_name = 'predictor/user_perfumes.html'
    context_object_name = 'perfumes'
    paginate_by = 5

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Perfume.objects.filter(added_by=user).order_by('-added_date')


def about(request):
    return render(request, 'predictor/about.html', {'title': 'about'})


class PerfumeDetailView(DetailView):
    model = Perfume


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



