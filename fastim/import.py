# a significant amount of code inspired or coppied form fastai.imports

# basics system relate libs

import io,os,operator,sys,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,random,time,math,bz2,typing,numbers,string
import multiprocessing,threading,urllib,tempfile,concurrent.futures,matplotlib,warnings,zipfile

from concurrent.futures import as_completed
from functools import partial,reduce
from itertools import starmap,dropwhile,takewhile,zip_longest
from copy import copy,deepcopy
from multiprocessing import Lock,Process,Queue,queues
from datetime import datetime
from contextlib import redirect_stdout,contextmanager
from collections.abc import Iterable,Iterator,Generator,Sequence
from typing import Union,Optional,TypeVar,Callable,Any
from types import SimpleNamespace
from pathlib import Path
from collections import OrderedDict,defaultdict,Counter,namedtuple
from enum import Enum,IntEnum
from textwrap import TextWrapper
from operator import itemgetter,attrgetter,methodcaller
from urllib.request import urlopen
from timeit import default_timer as timer 
from tqdm.auto import tqdm
import kaggle

# other basics libraries for EDA(mainly)
import numpy as np
from numpy import array,ndarray
import scipy as sp
from scipy import ndimage

import requests,yaml,matplotlib.pyplot as plt,pandas as pd, seaborn as sn
from pandas.api.types import is_categorical_dtype,is_numeric_dtype
from pdb import set_trace
from PIL import Image

# import all related to sklearn

from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import *


# import all related to fastai
import fastai
from fastai import *
from fastai.basics import *

from fastcore.all import *
from fastprogress.fastprogress import progress_bar,master_bar
from fastbook import *
import timm

from fastai.vision.all import *
from fastai.text.all import *
from fastai.tabular.all import *
from fastai.collab import *


# import all related to PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import *
from torchmetrics import *

from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision import datasets, transforms, models

# import pytorch lightning
import pytorch_lightning as pl

# import all related to HuggingFace
from transformers import *
from datasets import *

# coppied from fastai.basics

try:
    from types import WrapperDescriptorType,MethodWrapperType,MethodDescriptorType
except ImportError:
    WrapperDescriptorType = type(object.__init__)
    MethodWrapperType = type(object().__str__)
    MethodDescriptorType = type(str.join)
from types import BuiltinFunctionType,BuiltinMethodType,MethodType,FunctionType,LambdaType

pd.options.display.max_colwidth = 600
NoneType = type(None)
string_classes = (str,bytes)
mimetypes.init()

# PyTorch warnings
warnings.filterwarnings("ignore", message='.*nonzero.*', category=UserWarning)
warnings.filterwarnings("ignore", message='.*grid_sample.*', category=UserWarning)
warnings.filterwarnings("ignore", message='.*Distutils.*', category=UserWarning)

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable,Generator)) and getattr(o,'ndim',1)

def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    #Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, '__len__') and getattr(o,'ndim',1)

def all_equal(a,b):
    "Compares whether `a` and `b` are the same length and have the same contents"
    if not is_iter(b): return False
    return all(equals(a_,b_) for a_,b_ in itertools.zip_longest(a,b))

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
    return x

def one_is_instance(a, b, t): return isinstance(a,t) or isinstance(b,t)

def equals(a,b):
    "Compares `a` and `b` for equality; supports sublists, tensors and arrays too"
    if one_is_instance(a,b,type): return a==b
    if hasattr(a, '__array_eq__'): return a.__array_eq__(b)
    if hasattr(b, '__array_eq__'): return b.__array_eq__(a)
    cmp = (np.array_equal if one_is_instance(a, b, ndarray       ) else
           operator.eq    if one_is_instance(a, b, (str,dict,set)) else
           all_equal      if is_iter(a) or is_iter(b) else
           operator.eq)
    return cmp(a,b)


def pv(text, verbose):
    if verbose: print(text)
