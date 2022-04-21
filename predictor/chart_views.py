import pandas as pd
from django.views.generic import TemplateView

from users.models import Profile
from .models import Preference
from django.contrib.auth.models import User
from .charts import Chart, get_colors
from .recommendations import get_perfumes_and_reviews_df, build_model_for_user


class Dashboard(TemplateView):
    template_name = 'predictor/about.html'

    def get_context_data(self, **kwargs):
        # get the data from the default method
        context = super().get_context_data(**kwargs)

        # create a dataframe with all the records
        reviews_df = pd.DataFrame.from_records(Preference.objects.all().values())
        perfumes_df, df = get_perfumes_and_reviews_df(reviews_df)

        # df contains ['id'](perfume),['name'],['house'],['description'],['love'],['comments'], ['username']

        user_grouped_df = df.groupby('username', as_index=False)['love'].agg('mean')
        user_grouped_df['love'] = user_grouped_df['love'].multiply(100)

        house_grouped_df = df.groupby('house', as_index=False)['love'].agg('mean')
        house_grouped_df['love'] = house_grouped_df['love'].multiply(100)
        house_grouped_df.rename(columns={"love": "percent loved"})

        top_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).head(25)
        bottom_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).tail(25)

        love_perfumes_df = df.groupby('name', as_index=False)['love'].agg('sum')
        num_users = User.objects.all().count()
        love_perfumes_df['love'] = love_perfumes_df['love'].div(num_users)
        top_loved_df = love_perfumes_df[love_perfumes_df.love > 0].sort_values('love', ascending=False).head(10)

        user_reviews_counts_df = df.groupby('username', as_index=False)['id'].agg('count')

        user_loves_v_not_love_df = df.groupby('username', as_index=False)['love'].agg('sum')

        user_loves_v_not_love_df = pd.merge(left=user_reviews_counts_df, right=user_loves_v_not_love_df)
        user_loves_v_not_love_df['not_loves'] = user_loves_v_not_love_df['id'] - user_loves_v_not_love_df['love']

        # builds the accuracy data for the recommendation model
        multiNB_accuracy_df = pd.DataFrame(columns=['username', 'accuracy'])

        for user in User.objects.all():
            profile = Profile.objects.filter(user=user).first()
            if profile:
                user_reviews_df = profile.preference_dataframe()
                if len(user_reviews_df) > 0:
                    user_reviews_df.drop_duplicates(inplace=True)

                    classifier, accuracy, perfumes_df, perfume_reviews_df, counter = build_model_for_user(user_reviews_df)

                    temp_df = pd.DataFrame([[user.username, accuracy]], columns=['username', 'accuracy'])
                    multiNB_accuracy_df = pd.concat([multiNB_accuracy_df, temp_df], ignore_index=True)
                else:
                    continue
            else:
                continue

        # create a charts context to hold all of the charts
        context['charts'] = []

        context['table'] = user_loves_v_not_love_df.to_html()

        ### every chart is added the same way so I will just document the first one
        # create a chart object with a unique chart_id and color palette

        # if not chart_id or color palette is provided these will be randomly generated

        # the type of charts does need to be identified here and might differ from the chartjs type

        # city_payment_radar = Chart('radar', chart_id='city_payment_radar', palette=PALETTE)

        # # create a pandas pivot_table based on the fields and aggregation we want

        # # stacks are used for either grouping or stacking a certain column

        # city_payment_radar.from_df(df, values='total', stacks=['payment'], labels=['city'])

        # # add the presentation of the chart to the charts context

        # context['charts'].append(city_payment_radar.get_presentation())

        # exp_polar = Chart('polarArea', chart_id='polar01', palette=PALETTE)
        # exp_polar.from_df(df, values='total', labels=['payment'])
        # context['charts'].append(exp_polar.get_presentation())

        # perfume columns: 'id', 'name', 'house', 'description'
        # review columns: 'user'(object?), 'perfume' (object?), 'love'

        # exp_doughnut = Chart('doughnut', chart_id='doughnut01', palette=get_colors())
        # exp_doughnut.from_df(df, values='love', labels=['house'])
        # context['charts'].append(exp_doughnut.get_presentation())

        # df contains ['id'](perfume),['name'],['house'],['description'],['love'],['comments'], ['username']

        exp_bar = Chart('bar', chart_id='bar01', palette=get_colors())
        exp_bar.from_df(user_grouped_df, 'x-axis', 'y-axis', 'Percent of Total Perfumes Loved by Each User',
                        values='love', labels=['username'])
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

        # exp5_bar = Chart('stackedBar', chart_id='bar05', palette=get_colors())
        # exp5_bar.from_df(user_loves_v_not_love_df, 'x-axis', 'y-axis',
        #                  'Number of Loved vs Not-Loved Reviews by Each User',
        #                  values='id', stacks=['love'], labels=['username'])
        # context['charts'].append(exp5_bar.get_presentation())

        exp4_bar = Chart('bar', chart_id='bar04', palette=get_colors())
        exp4_bar.from_df(multiNB_accuracy_df, 'x-axis', 'y-axis', 'Recommendation Accuracy of NB Model for Each User',
                         values='accuracy', labels=['username'])
        context['charts'].append(exp4_bar.get_presentation())

        # exp_doughnut = Chart('groupedBar', chart_id='doughnut01', palette=get_colors())
        # exp_doughnut.from_df(df, values='love', stacks=['username'], labels=[''])
        # context['charts'].append(exp_doughnut.get_presentation())
        #
        # exp_grouped_bar = Chart('groupedBar', chart_id='groupedbar01', palette=get_colors())
        # exp_grouped_bar.from_df(df, values='love', stacks=['username'], labels=['house'])
        # context['charts'].append(exp_grouped_bar.get_presentation())
        #
        # exp_stacked_bar = Chart('stackedBar', chart_id='stacked01', palette=get_colors())
        # exp_stacked_bar.from_df(df, values='love', stacks=['username'], labels=['house'])
        # context['charts'].append(exp_stacked_bar.get_presentation())
        #
        # city_gender_h = Chart('stackedHorizontalBar', chart_id='city_gender_h', palette=PALETTE)
        # city_gender_h.from_df(df, values='total', stacks=['gender'], labels=['city'])
        # context['charts'].append(city_gender_h.get_presentation())

        # city_payment = Chart('groupedBar', chart_id='city_payment', palette=PALETTE)
        # city_payment.from_df(df, values='total', stacks=['payment'], labels=['date'])
        # context['charts'].append(city_payment.get_presentation())
        #
        # city_payment_h = Chart('horizontalBar', chart_id='city_payment_h', palette=PALETTE)
        # city_payment_h.from_df(df, values='total', stacks=['payment'], labels=['city'])
        # context['charts'].append(city_payment_h.get_presentation())
        #

        #
        # city_gender = Chart('stackedBar', chart_id='city_gender', palette=PALETTE)
        # city_gender.from_df(df, values='total', stacks=['gender'], labels=['city'])
        # context['charts'].append(city_gender.get_presentation())

        return context
