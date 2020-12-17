import math
import random
import torch
from flask import Markup
from IPython.display import HTML as html_print

def generate_activations(model, vocab, text):
    model.eval()
    with torch.no_grad():
        text_array = vocab.words_to_array(text)

        hidden = None
        activation_values = []
        for sample in text_array:
            output, hidden = model.inference(sample, hidden)
            activation_val = hidden[0].cpu().detach().numpy().flatten()
            activation_values.append(activation_val)
    return activation_values

# get html element
def cstr(s, color='black'):
    if (s == ' '):
        return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
    elif (s == '\n'):
        return "<text style=color:#000;padding-left:10px;background-color:{}> <br></text>".format(color, s)
    else:
        return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

# get appropriate color for value
def get_clr(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
        '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
        '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
        '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']

    value_idx = math.floor(value * len(colors))
    if value_idx >= len(colors):
        value_idx = len(colors) - 1

    return colors[value_idx]

def visualize(generated_sentence, activation_values, cell_no):
    text_colours = []

    for char, val in zip(generated_sentence, activation_values):
        text_clr = (char, get_clr(val[cell_no]))
        text_colours.append(text_clr)

    html_colors = html_print(''.join([cstr(ti, color=ci) for ti,ci in text_colours]))
    return html_colors

def construct_visualization(activation_values, text, cells, n_vis=4):
    result = []
    
    if cells is None:
        cells = random.sample(range(len(activation_values[0])), n_vis)
    for cell_no in cells:
        print("Processing cell_no: " + str(cell_no))
        html_colors = visualize(text, activation_values, cell_no)
        cell_result = html_colors.data
        result.append((cell_no, Markup(cell_result)))
    return result

def generate_cell_visualization(language_model, vocab, generated_sentence, device="cpu", cells=None, n_vis=4):
    activation_values = generate_activations(language_model, vocab, generated_sentence)
    visualization = construct_visualization(activation_values, generated_sentence, cells, n_vis)

    return visualization