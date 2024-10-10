import pandas as pd
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import base64
import os
from wordcloud import WordCloud
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

csv_1 = os.path.join(THIS_FOLDER, 'Sentiment_English.csv')
csv_2 = os.path.join(THIS_FOLDER, 'Top_Word.csv')

df1= pd.read_csv(csv_1)
df2 = pd.read_csv(csv_2)

Ulasan = df1['Ulasan']

app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
app.title='Customer Feedback Sentiment Analysis'
server = app.server

CONTENT_STYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'padding': '20px 10p'
}


##className="text-success text-center ms-1"##

title = dbc.Row([
        html.H1('Customer Feedback Sentiment Analysis Dashboard',className="text-success text-center ms-1")
    ])

cards_data = [
    {"title": "Total Positive", "value": "319"},
    {"title": "Total Negative", "value": "57"},
    {"title": "Total Neutral", "value": "24"},
    {"title": "Total Ulasan", "value": "400"},
]

cards = []
for card in cards_data:
    cards.append(
        dbc.Card(
            dbc.CardBody([
                html.H6(card['title'], className="card-title", style={'text-align': 'center'}),
                html.H3(card['value'], className="card-value", style={'text-align': 'center', 'color': '#28a745'}),
            ]),
            style={"width": "8rem", "margin": "5px"},
        )
    )

Identifi_data =  dbc.Container(
    dbc.Row(
        [dbc.Col(card, width="auto") for card in cards],  # Menggunakan kolom untuk responsivitas
        justify="center"  # Menyusun card di tengah
    ),
    style={"padding-top": "3px"}
)


import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')

selected_columns = ["Ulasan", "clean", "Vader_Sentiment"]
fig_table = go.Figure(data=[go.Table(
    header=dict(values=selected_columns,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df1[col] for col in selected_columns],
               fill_color='lavender',
               align='left'))
])

##html.H4('Dataset Customer Feedback', className='card-title', style={'text-align': 'center'}),
                   ## html.Div([
                        #dash_table.DataTable(
                    #                    id='table',
                                     #   columns=[{"name": i, "id": i} for i in df1.columns],
                                      #  data=df1.to_dict('records'),
                                       # style_table={'height': '300px', 'overflowY': 'auto'},
                                       #style_cell={'textAlign': 'left'},
                                      #  page_size=10
                                 #   )
                   # ])##

# Data for bar chart
data = df2[:10]
data = data.sort_values(by='frequency', ascending=False)
fig_bar = px.bar(data, x='word', y='frequency', text='word', color='word',
                 color_discrete_sequence=px.colors.sequential.Viridis, title='Rank of word',
                 template='simple_white')

fig_bar.update_layout(
     title={
        'text': 'Rank of word',   # Teks judul
        'y': 0.9,                 # Posisi judul pada sumbu y (0 sampai 1)
        'x': 0.5,                 # Posisi judul pada sumbu x (0.5 untuk tengah)
        'xanchor': 'center',      # Anchor di tengah secara horizontal
        'yanchor': 'top',         # Anchor di atas secara vertikal
        'font': {
            'size': 20,           # Ukuran font
            'color': 'black',     # Warna teks judul
            'family': 'Arial',    # Jenis font (opsional)
            'weight': 'bold'      # Mengatur teks tebal
        }
    },
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
)
fig_bar.update_traces(textposition='inside', textfont_size=11)


def Word_Cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(text)
    # Simpan gambar ke objek BytesIO
    img = io.BytesIO()
    plt.figure(figsize=(8, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud Visualization", fontsize=16, loc='center', pad=20)
    plt.savefig(img, format="png", bbox_inches="tight")
    plt.close()
    img.seek(0)
    # Encode gambar menjadi base64
    encoded_image = base64.b64encode(img.getvalue()).decode('utf-8')
    return encoded_image

wordcloud = df2.set_index('word').to_dict()['frequency']
wordcloud_image = Word_Cloud(wordcloud)

#Card Visualisasi
visualisasi = dbc.Container([
    dbc.Row([
        # Card 1: Tabel
        dbc.Col([
            dbc.Card([
                dbc.CardBody([               
                       dcc.Graph(
                               figure=fig_table,
                       )
                ])
            ])
        ], width=4),
        
        # Card 2: Top Word
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_bar)
                ])
            ])
        ], width=4),
        
        # Card 3: Word Cloud
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Img(src=f'data:image/png;base64,{wordcloud_image}', style={"width": "65%", "height": "100%"})
                ], style={"width": "50rem", "height": "25rem"})
            ])
        ], width=4)
    ],className="g-4")
], fluid=True)


sentiment = df1["Vader_Sentiment"].value_counts()
fig = px.pie(values=sentiment.values,
             names=sentiment.index,
             color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_traces(textposition='inside',
                  textfont_size=11,
                  textinfo='percent+label')
fig.update_layout(title={
                     'text': 'Customer Feedback Sentiment',   # Teks judul
                      'y': 1.0,                 # Posisi judul pada sumbu y (0 sampai 1)
                      'x': 0.5,                 # Posisi judul pada sumbu x (0.5 untuk tengah)
                      'xanchor': 'center',      # Anchor di tengah secara horizontal
                       'yanchor': 'top',         # Anchor di atas secara vertikal
                      'font': {
                         'size': 20,           # Ukuran font
                         'color': 'black',     # Warna teks judul
                         'family': 'Arial',    # Jenis font (opsional)
                         'weight': 'bold'      # Mengatur teks tebal
                          }
                      },
                  uniformtext_minsize=12,
                  uniformtext_mode='hide')

csv_3 = os.path.join(THIS_FOLDER, 'Positive.csv')
csv_4 = os.path.join(THIS_FOLDER, 'Neutral.csv')
csv_5 = os.path.join(THIS_FOLDER, 'Negative.csv')

positive = pd.read_csv(csv_3)
neutral = pd.read_csv(csv_4)
negative = pd.read_csv(csv_5)

import plotly.graph_objects as go
fig_word = go.Figure(data=[
    go.Bar(name='positive', x=positive['Term'][:7].values, y=positive['Frequency'][:7].values,text=positive['Frequency'][:7].values,marker_color='red'),
    go.Bar(name='neutral', x=neutral['Term'][:7].values, y=neutral['Frequency'][:7].values,text=positive['Frequency'][:7].values,marker_color='yellow', ),
    go.Bar(name='negative', x=negative['Term'][:7].values, y=negative['Frequency'][:7].values,text=positive['Frequency'][:7].values,marker_color='blue',)
])
fig_word.update_layout(barmode='stack', xaxis_tickangle=-45,
                       title={
                        'text': 'Sentiment Score by Word',   # Teks judul
                        'y': 0.9,                 # Posisi judul pada sumbu y (0 sampai 1)
                        'x': 0.5,                 # Posisi judul pada sumbu x (0.5 untuk tengah)
                         'xanchor': 'center',      # Anchor di tengah secara horizontal
                         'yanchor': 'top',         # Anchor di atas secara vertikal
                         'font': {
                         'size': 20,           # Ukuran font
                         'color': 'black',     # Warna teks judul
                         'family': 'Arial',    # Jenis font (opsional)
                         'weight': 'bold'      # Mengatur teks tebal
                          }
                      }, 
                       template='simple_white')


#Card Visualisasi2
visualisasi2 = dbc.Container([
    dbc.Row([
        # Card 4: Diagram Lingkaran 
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ])
        ], style={'display': 'inline-block', 'width': '48%', 'margin-right': '2%'}),
        
        # Card 5: Sentiment by word
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig_word)
                ])
            ])
        ], style={'display': 'inline-block', 'width': '48%'})
    ],className="g-4")
], fluid=True)

content = html.Div(
    [
       title    ,
       html.Hr(),
       Identifi_data,
       html.Br(),
       visualisasi,
       html.Br(),
       visualisasi2, 
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([content])


if __name__ == '__main__':
    app.run(debug=True)
