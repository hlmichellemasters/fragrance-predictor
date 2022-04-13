from datetime import datetime

from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from users.models import Profile
from .forms import ReviewForm
from .models import Perfume, Preference

import pandas as pd


def home(request):
    context = {
        'perfumes': Perfume.objects.all()
    }

    return render(request, 'predictor/perfume_list.html', context)


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


def about(request):
    return render(request, 'predictor/about.html', {'title': 'about'})


# class PerfumeDetailView(DetailView):
#     model = Perfume


# class PerfumeDetailView(DetailView):
#     model = Perfume
#     template_name = 'predictor/perfume_detail_list.html'
#     context_object_name = 'perfume'
#     paginate_by = 5
#
#     def get_queryset(self):
#         perfume = get_object_or_404(Perfume, name=self.kwargs.get('name'))
#         return Perfume.objects.filter(name=perfume.name)


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


# @login_required
# def preferences(request):
#     if request.method == "POST":
#         pref_form = ReviewForm(request.POST, instance=request.user)
#
#         if pref_form.is_valid():
#             pref_form.save()
#             messages.success(request, f'Awesome! Your preferences have been updated :)')
#             return redirect('profile')
#
#     else:
#         pref_form = ReviewForm(instance=request.user)
#
#     context = {
#         'pref_form': pref_form,
#     }
#
#     return render(request, 'users/preference_form.html', context)


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


# def user_preference_list(request):
#     latest_preference_list = Preference.objects.order_by('-modified_date')[:9]
#     context = {'latest_preference_list': latest_preference_list}
#     return render(request, 'user_preference_list.html', context)


# def preference_detail(request, preference_id):
#     review = get_object_or_404(Preference, pk=preference_id)
#     return render(request, 'predictor/preference_detail.html', {'review': review})

@login_required()
def add_preference(request, pk):
    perfume = get_object_or_404(Perfume, pk=pk)

    if request.POST:
        form = ReviewForm(request.POST)
    else:
        form = ReviewForm()

    if form.is_valid():
        # love = form.cleaned.data['love']
        # comment = form.cleaned.data['comment']
        # user_name = form.cleaned.data['user']
        # review.love = love
        # review.comment = comment
        review = form.save(commit=False)
        review.perfume = perfume
        review.user = request.user
        review.modified_date = datetime.now()
        review.save()
        return HttpResponseRedirect(reverse('perfume-detail', args=(pk,)))

    return render(request, 'predictor/perfume-detail', {'perfume': perfume, 'form': form})


# showcasing recommendations just for michellem user first
@login_required
def user_recommendation_list(request):
    # hard-coding just default user for now
    user = User.objects.filter(username='michellem').first()

    # gets the profile of the user
    profile = Profile.objects.filter(user=user).first()
    reviews_df = profile.preference_dataframe()
    perfumes_df = pd.DataFrame.from_records(Perfume.objects.all().values('id', 'name', 'house', 'description'))

    # inner join the perfume and review data to include all important columns for reviewed perfumes
    perfume_reviews_df = pd.merge(reviews_df, perfumes_df, how='inner', left_on='perfume_id', right_on='id')

    # to display the data in the template
    html_data = perfume_reviews_df.to_html()

    train_data, test_data, train_labels, test_labels = \
        train_test_split(perfume_reviews_df['description'].values.astype('U'), perfume_reviews_df['love'],
                         test_size=0.2, random_state=1)

    counter = CountVectorizer(stop_words='english')
    counter.fit(train_data)
    train_counts = counter.transform(train_data)
    test_counts = counter.transform(test_data)

    length_train = len(train_data)
    length_test = len(test_data)
    #
    # classifier = MultinomialNB()
    # classifier.fit(train_counts, train_labels)
    #
    #
    # predictions = classifier.predict(test_counts)
    #
    # print("Accuracy score: " + str(accuracy_score(test_labels, predictions)))
    return render(request, 'predictor/user_recommendation_list.html', {'username': request.user.username,
                                                                       'reviews': reviews_df,
                                                                       'lengthDF': len(reviews_df),
                                                                       'lengthTrain': len(train_data),
                                                                       'lengthTest': len(test_data),
                                                                       'perfumes': perfumes_df,
                                                                       'perfumeReviews': html_data
                                                                       })


