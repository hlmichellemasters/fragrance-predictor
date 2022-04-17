from django import forms
from .models import Preference, Perfume


# form for adding a review onto a perfume detail page
class ReviewForm(forms.ModelForm):
    class Meta:
        model = Preference
        fields = ['love', 'comment']
        widgets = {
            'comment': forms.Textarea(attrs={'cols': 40, 'rows': 5})
        }


# form for the cold-start recommendation
class RecommendationForm(forms.Form):
    perfume_loves = forms.ModelMultipleChoiceField(required=False, label="Perfumes you love:", widget=GroupedSelect(), queryset=Perfume.objects.all())



