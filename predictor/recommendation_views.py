from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
import math
from datetime import datetime

import pandas as pd

from users.models import Profile
from .forms import ReviewForm, RecommendationForm
from .models import Perfume, Preference
from . import recommendations as rec
from .recommendations import find_perfumes_from_features


@login_required()
def add_preference(request, pk):
    perfume = get_object_or_404(Perfume, pk=pk)

    if request.POST:
        form = ReviewForm(request.POST)
    else:
        form = ReviewForm()

    if form.is_valid():
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
    user = request.user

    # gets the profile and reviews of the user
    profile = Profile.objects.filter(user=user).first()
    user_reviews_df = profile.preference_dataframe()
    if len(user_reviews_df) < 1:
        return render(request, 'predictor/user_recommendation_list.html', {'username': request.user.username,
                                                                           'accuracyScore': 0})

    # build the model for the user
    classifier, accuracy, perfumes_df, perfume_reviews_df, counter = rec.build_model_for_user(user_reviews_df)

    # extract all perfumes that user hasn't reviewed yet into its own dataframe
    unreviewed_perfumes_df = perfumes_df[~perfumes_df.id.isin(perfume_reviews_df.perfume_id)]
    # transform the unreviewed perfume data for classifier
    unreviewed_perfumes_data = counter.transform(unreviewed_perfumes_df['features'].values.astype('U'))
    # estimate probabilities for remaining perfumes (that user hasn't yet reviewed)
    unreviewed_perfumes_df['love_probability'] = classifier.predict_proba(unreviewed_perfumes_data)[:, 1]
    # sort them and take the top 10?
    sorted_recommendations_df = unreviewed_perfumes_df.sort_values(by=['love_probability'], ascending=False)
    top_10_recommendations_df = sorted_recommendations_df.head(10)
    # compile a list of the top 10 to send to display
    recommended_perfumes = []
    percent_confidence = []
    for i in range(0, top_10_recommendations_df.shape[0]):
        perfume_id = top_10_recommendations_df.iloc[i].at['id']
        this_percent = math.floor((top_10_recommendations_df.iloc[i].at['love_probability']) * 100)
        percent_confidence.append(this_percent)
        top_perfume = Perfume.objects.get(pk=perfume_id)
        recommended_perfumes.append(top_perfume)
        perfumes = zip(recommended_perfumes, percent_confidence)

    return render(request, 'predictor/user_recommendation_list.html', {'username': request.user.username,
                                                                       'accuracyScore': accuracy,
                                                                       'perfumes': perfumes,
                                                                       'percent-confidence': percent_confidence,
                                                                       })


def recommendation_form(request):
    if request.POST:
        form = RecommendationForm(request.POST)
        if form.is_valid():
            notes_loves = form.cleaned_data['other_notes_loves']
            notes_not_loves = form.cleaned_data['other_notes_not_loves']
            perfume_loves = form.cleaned_data['perfume_loves']
            perfume_not_loves = form.cleaned_data['perfume_not_loves']

            if perfume_loves:
                loves_df = pd.DataFrame([p.__dict__ for p in perfume_loves])
            else:
                loves_df = pd.DataFrame()
            if perfume_not_loves:
                not_loves_df = pd.DataFrame([p.__dict__ for p in perfume_not_loves])
            else:
                not_loves_df = pd.DataFrame()

            perfumes = find_perfumes_from_features(notes_loves, notes_not_loves, loves_df, not_loves_df)

            # table = DataFrame.to_html(not_loves_df)

            return render(request, 'predictor/recommendation_list.html', {
                                                                          # 'table': table,
                                                                          'form': form,
                                                                          'perfumes': perfumes,
                                                                          'loves_df': loves_df,
                                                                          'not_loves_df': not_loves_df,
                                                                          'perfume_loves': perfume_loves,
                                                                          'perfume_not_loves': perfume_not_loves,
                                                                          'notes_loves': notes_loves,
                                                                          'notes_not_loves': notes_not_loves})

    else:
        form = RecommendationForm()

    return render(request, 'predictor/recommendation_form.html', {'form': form})


def recommendation_list(request):
    # perfume_loves = request.perfume_loves
    # perfume_not_loves = request.perfume_not_loves
    # notes_loves = request.other_notes_loves
    # notes_not_loves = request.other_notes_not_loves
    #
    # loves_df = DataFrame.from_records(perfume_loves)
    # not_loves_df = DataFrame.from_records(perfume_not_loves)
    #
    # perfumes = find_perfumes_from_features(notes_loves, notes_not_loves, loves_df, not_loves_df)
    # table = DataFrame.to_html(perfumes)

    return render(request, 'predictor/recommendation_list.html',
                  # {'table': table,
                  #                                                 'perfumes': perfumes,
                  #                                                 'perfume_loves': perfume_loves,
                  #                                                 'perfume_not_loves': perfume_not_loves,
                  #                                                 'notes_loves': notes_loves,
                  #                                                 'notes_not_loves': notes_not_loves}
                  )
