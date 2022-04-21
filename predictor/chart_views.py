import pandas as pd
from django.views.generic import TemplateView
from .models import Preference
from .charts import Chart, get_colors
from .recommendations import get_perfumes_and_reviews_df


class Dashboard(TemplateView):
    template_name = 'predictor/about.html'

    def get_context_data(self, **kwargs):

        # get the data from the default method
        context = super().get_context_data(**kwargs)

        # create a dataframe with all the records
        reviews_df = pd.DataFrame.from_records(Preference.objects.all().values())
        perfumes_df, df = get_perfumes_and_reviews_df(reviews_df)

        # df contains ['id'](perfume),['name'],['house'],['description'],['love'],['comments'], ['username']

        # df = merged_df.groupby('transaction_id', as_index=False)['price'].agg('sum')
        user_grouped_df = df.groupby('username', as_index=False)['love'].agg('mean')
        house_grouped_df = df.groupby('house', as_index=False)['love'].agg('mean')
        top_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).head(25)
        bottom_house_grouped_df = house_grouped_df.sort_values('love', ascending=False).tail(25)
        love_perfumes_df = df.groupby('name', as_index=False)['love'].agg('sum')
        top_loved_df = love_perfumes_df[love_perfumes_df.love > 0].sort_values('love', ascending=False).head(10)
        # create a charts context to hold all of the charts
        context['charts'] = []

        context['table'] = top_loved_df.to_html()

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
        exp_bar.from_df(user_grouped_df, values='love', labels=['username'])
        context['charts'].append(exp_bar.get_presentation())

        exp2_bar = Chart('bar', chart_id='bar02', palette=get_colors())
        exp2_bar.from_df(top_loved_df, values='love', labels=['name'])
        context['charts'].append(exp2_bar.get_presentation())

        exp_horizontal_bar = Chart('horizontalBar', chart_id='bar03', palette=get_colors())
        exp_horizontal_bar.from_df(top_house_grouped_df, values='love', labels=['house'])
        context['charts'].append(exp_horizontal_bar.get_presentation())

        exp2_horizontal_bar = Chart('horizontalBar', chart_id='bar04', palette=get_colors())
        exp2_horizontal_bar.from_df(bottom_house_grouped_df, values='love', labels=['house'])
        context['charts'].append(exp2_horizontal_bar.get_presentation())

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
