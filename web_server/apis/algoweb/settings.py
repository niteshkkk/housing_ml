"""
Django settings for algoweb project.

Generated by 'django-admin startproject' using Django 1.9.9.

For more information on this file, see
https://docs.djangoproject.com/en/1.9/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.9/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.9/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '%l=tvh5o(w=#@)x4n6_w8c6kwqckkjw)61x%nd6u88*94=d%6)'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

# INSTALLED_APPS = [
#   'django.contrib.admin',
#  'django.contrib.auth',
# 'django.contrib.contenttypes',
#  'django.contrib.sessions',
#  'django.contrib.messages',
#  'django.contrib.staticfiles',
#]
INSTALLED_APPS = (
    'django.contrib.contenttypes',

    # Django Suit skin for admin panel
    'suit',

    # Django contrib
    'django.contrib.auth',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.admin',
    'django.contrib.humanize',


    # REST APIs
    'rest_framework',
    'rest_framework_swagger',
    'rest_framework.authtoken',

    # Utilities
    'django_extensions',
    'sekizai',
    'jsonify',

    # My apps
    'mpld3',
    # 'tinymce',
)

# MIDDLEWARE_CLASSES = [
#   'django.middleware.security.SecurityMiddleware',
#  'django.contrib.sessions.middleware.SessionMiddleware',
# 'django.middleware.common.CommonMiddleware',
#'django.middleware.csrf.CsrfViewMiddleware',
#'django.contrib.auth.middleware.AuthenticationMiddleware',
# 'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
#'django.contrib.messages.middleware.MessageMiddleware',
#'django.middleware.clickjacking.XFrameOptionsMiddleware',
#]

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # Uncomment the next line for simple clickjacking protection:
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_tools.middlewares.ThreadLocal.ThreadLocalMiddleware',
    #'web_backend.middleware.HttpRequestAuthFilter',
)

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest.LimitOffsetPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_FILTER_BACKENDS': (
        'rest_framework_filters.backends.DjangoFilterBackend',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ),
    'DATETIME_FORMAT': "%Y-%m-%d %H:%M:%S",
}

ROOT_URLCONF = 'urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.9/ref/settings/#databases


# Password validation
# https://docs.djangoproject.com/en/1.9/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

mpld3_dir = os.path.join(os.path.dirname(BASE_DIR), "mpld3_html")

STATICFILES_DIRS = [os.path.abspath(mpld3_dir)]

print(STATICFILES_DIRS)


# Internationalization
# https://docs.djangoproject.com/en/1.9/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.9/howto/static-files/

STATIC_URL = '/static/'

from logger_settings import *  # noqa