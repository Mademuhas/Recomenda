from django.shortcuts import render
from django.http import HttpResponse
import json
from recomenda.functions import (
    get_vinho, get_the_json,
)

from django.views.decorators.csrf import csrf_exempt





def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def set_vinho (request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    content = body['content']
    obj = get_the_json (content)
    result = get_vinho(obj)
    return HttpResponse(result)

