from django import forms
from .models import Preference, Perfume
from itertools import groupby
from django.forms.models import ModelChoiceIterator, ModelMultipleChoiceField


# from https://blog.mounirmesselmeni.de/2013/11/25/
# django-grouped-select-field-for-modelchoicefield-or-modelmultiplechoicefield/
class GroupedMultipleModelChoiceField(ModelMultipleChoiceField):

    def __init__(self, group_by_field, group_label=None, *args, **kwargs):
        """
        group_by_field is the name of a field on the model
        group_label is a function to return a label for each choice group
        """
        super(GroupedMultipleModelChoiceField, self).__init__(*args, **kwargs)
        self.group_by_field = group_by_field
        if group_label is None:
            self.group_label = lambda group: group
        else:
            self.group_label = group_label

    def _get_choices(self):
        """
        Exactly as per ModelChoiceField except returns new iterator class
        """
        if hasattr(self, '_choices'):
            return self._choices
        return GroupedModelChoiceIterator(self)
    choices = property(_get_choices, ModelMultipleChoiceField._set_choices)


class GroupedModelChoiceIterator(ModelChoiceIterator):

    def __iter__(self):
        if self.field.empty_label is not None:
            yield u"", self.field.empty_label
        else:
            for group, choices in groupby(
                    self.queryset.all(),
                    key=lambda row: getattr(
                        row, self.field.group_by_field)):
                if group is not None:
                    yield (
                        self.field.group_label(group),
                        [self.choice(ch) for ch in choices])


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
    perfume_loves = GroupedMultipleModelChoiceField(required=False, label="Perfumes you love:",
                                                    group_by_field='house',
                                                    help_text="hold cmd/ctrl to select multiple",
                                                    queryset=Perfume.objects.all().order_by('house', 'name'))
    perfume_loves.widget.attrs.update(size='15')

    other_notes_loves = forms.CharField(required=False, label="Notes you love:")

    perfume_not_loves = GroupedMultipleModelChoiceField(required=False, label="Perfumes you don't love:",
                                                        group_by_field='house',
                                                        help_text="hold cmd/ctrl to select multiple",
                                                        queryset=Perfume.objects.all().order_by('house', 'name'))
    perfume_not_loves.widget.attrs.update(size='15')

    other_notes_not_loves = forms.CharField(required=False, label="Notes you don't love:")




