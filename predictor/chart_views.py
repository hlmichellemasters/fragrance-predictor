import pandas as pd
from django.views.generic import TemplateView
from users.models import Profile
from .models import Preference
from django.contrib.auth.models import User
from .charts import Chart, get_colors
from .recommendations import get_perfumes_and_reviews_df, build_model_for_user


# Subclass of a TemplateView to create the dashboard of charts on the About Page
class Dashboard(TemplateView):
    template_name = 'predictor/about.html'

    def get_context_data(self, **kwargs):
        # get the data from the default method
        context = super().get_context_data(**kwargs)

        # create a dataframe with all the records
        reviews_df = pd.DataFrame.from_records(Preference.objects.all().values())
        perfumes_df, df = get_perfumes_and_reviews_df(reviews_df)

        # df contains ['id'](perfume),['name'],['house'],['description'],['love'],['comments'], ['username']
        # groups the reviews dataframe by user and gathers the mean for the love column
        user_grouped_df = df.groupby('username', as_index=False)['love'].agg('mean')
        # turns the love average into an easier to understand "percentage"
        user_grouped_df['love'] = user_grouped_df['love'].multiply(100)
        # renames the column to elucidate the meaning of the data
        user_grouped_df.rename(columns={"love": "percent loved"})

        # groups by the house (or brand), and grabs the mean of the love column
        house_grouped_df = df.groupby('house', as_index=False)['love'].agg('mean')
        house_grouped_df['love'] = house_grouped_df['love'].multiply(100)
        house_grouped_df.rename(columns={"love": "percent loved"})
        # sorts the house aggregated data by the most loves and takes the top 25 for clarity of the chart
        top_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).head(25)
        # optional bottom 25 houses (but none have any "loves"), all 0%
        # bottom_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).tail(25)

        # groups by the perfume and then sums the number times it was loved
        love_perfumes_df = df.groupby('name', as_index=False)['love'].agg('sum')
        # gets count of number of users in database
        num_users = User.objects.all().count()
        # divides the sum of the times the perfume was loved and then divides by the number of users (percent loved)
        love_perfumes_df['love'] = love_perfumes_df['love'].div(num_users)
        # sorts by the most loved and then grabs the top 25
        top_loved_df = love_perfumes_df[love_perfumes_df.love > 0].sort_values('love', ascending=False).head(20)
        top_loved_df['love'] = top_loved_df['love'].multiply(100)
        top_loved_df.rename(columns={"love": "percent loved"})

        user_reviews_counts_df = df.groupby('username', as_index=False)['id'].agg('count')

        user_loves_v_not_love_df = df.groupby('username', as_index=False)['love'].agg('sum')

        user_loves_v_not_love_df = pd.merge(left=user_reviews_counts_df, right=user_loves_v_not_love_df)
        user_loves_v_not_love_df['not_loves'] = user_loves_v_not_love_df['id'] - user_loves_v_not_love_df['love']

        # builds the accuracy data for the recommendation model
        multiNB_accuracy_df = pd.DataFrame(columns=['username', 'accuracy'])

        # for each user that has a profile and has reviews
        for user in User.objects.all():
            profile = Profile.objects.filter(user=user).first()
            if profile:
                user_reviews_df = profile.preference_dataframe()
                if len(user_reviews_df) > 0:
                    # drop any duplicates in their reviews
                    user_reviews_df.drop_duplicates(inplace=True)
                    # build a NB classifier model based on the user's reviews and receive back the accuracy of model
                    classifier, accuracy, perfumes_df, perfume_reviews_df, counter = \
                        build_model_for_user(user_reviews_df)
                    # build a dataframe  to append
                    temp_df = pd.DataFrame([[user.username, accuracy]], columns=['username', 'accuracy'])
                    multiNB_accuracy_df = pd.concat([multiNB_accuracy_df, temp_df], ignore_index=True)
                else:
                    continue
            else:
                continue
        multiNB_accuracy_df['accuracy'] = multiNB_accuracy_df['accuracy'].multiply(100)

        # create a charts context to hold all of the charts
        context['charts'] = []

        # table used for testing the data being passed to the charts
        # context['table'] = user_loves_v_not_love_df.to_html()

        # df contains ['id'](perfume),['name'],['house'],['description'],['love'],['comments'], ['username']

        # creates a chart object and specifies the type, id and assigns color palette
        exp_bar = Chart('bar', chart_id='bar01', palette=get_colors())
        # assigns the data, title, and values and labels to the chart
        exp_bar.from_df(user_grouped_df, 'x-axis', 'y-axis', 'Percent of Total Perfumes Loved by Each User',
                        values='love', labels=['username'])
        # adds the chart to the context variable to be passed to the template
        context['charts'].append(exp_bar.get_presentation())

        exp2_bar = Chart('bar', chart_id='bar02', palette=get_colors())
        exp2_bar.from_df(top_loved_df, 'x-axis', 'y-axis', 'Percent of Users that Loved the Most Popular Perfumes',
                         values='love', labels=['name'])
        context['charts'].append(exp2_bar.get_presentation())

        exp_horizontal_bar = Chart('horizontalBar', chart_id='horizontalBar', palette=get_colors())
        exp_horizontal_bar.from_df(top_house_grouped_df, 'x-axis', 'y-axis',
                                   'Percent of Perfumes Loved by Users of Most Popular Houses', values='love',
                                   labels=['house'])
        context['charts'].append(exp_horizontal_bar.get_presentation())

        exp3_bar = Chart('bar', chart_id='bar03', palette=get_colors())
        exp3_bar.from_df(user_reviews_counts_df, 'x-axis', 'y-axis', 'Number of Reviews by Each User',
                         values='id', labels=['username'])
        context['charts'].append(exp3_bar.get_presentation())

        exp4_bar = Chart('bar', chart_id='bar04', palette=get_colors())
        exp4_bar.from_df(multiNB_accuracy_df, 'x-axis', 'y-axis', 'Recommendation Accuracy of NB Model for Each User',
                         values='accuracy', labels=['username'])
        context['charts'].append(exp4_bar.get_presentation())

        return context
