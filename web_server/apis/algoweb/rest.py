from rest_framework.pagination import LimitOffsetPagination as DrfLimitOffsetPagination
from rest_framework.response import Response
from collections import OrderedDict


class LimitOffsetPagination(DrfLimitOffsetPagination):
    def get_paginated_response(self, data):
        meta = {
            'count': self.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
        }
        return Response(OrderedDict([
            ('meta', meta),
            ('results', data)
        ]))
