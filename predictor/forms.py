from django import forms
from .models import Preference


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Preference
        fields = ['love', 'comment']
        widgets = {
            'comment': forms.Textarea(attrs={'cols': 40, 'rows': 5})
        }




