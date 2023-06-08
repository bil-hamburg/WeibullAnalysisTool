#Using Python version: 3.7.10
#
import base64
import os
from urllib.parse import quote as urlquote
import pandas as pd
import time
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
from datetime import datetime
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm, Inches, Mm, Emu
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import io
import matplotlib
#https://community.plotly.com/t/upload-csv-with-dcc-upload-to-store-using-dcc-store-and-then-displaying-table/67802/3
UPLOAD_DIRECTORY = "/WeibullAnalyse"
Datetimenow = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))


app = dash.Dash(__name__)
server = app.server

df_CSV = pd.DataFrame(
    [
        ["Ausgefallen", 80],
        ["Ausgefallen", 90],
        ["Ausgefallen", 120],
        ["In Betrieb", 100],
        ["In Betrieb", 110],
        ["Ausgefallen", 140],
        ["Ausgefallen", 50],
        ["Ausgefallen", 60],
        ["Ausgefallen", 130],
        ["Ausgefallen", 150],
        ["In Betrieb", 40],
        ["In Betrieb", 70],
        ["In Betrieb", 75],
    ],
    columns=["AusgefallenOderInBetrieb", "Laufzeit"],
)



dftest = pd.DataFrame(dict(
    x=[1, 3, 2, 4],
    y=[1, 2, 3, 4]
))

figure_zuverlässigkeit = px.line(dftest, x="x", y="y", title="Unsorted Input")
figure_Ausfallrate = px.line(dftest, x="x", y="y", title="Unsorted Input")
figure_Wahrscheinlichkeitsverteilungsfunktion = px.line(dftest, x="x", y="y", title="Unsorted Input")
figure_KumulativeVerteillungsfunktion = px.line(dftest, x="x", y="y", title="Unsorted Input")



app.layout = html.Div(
    [
        html.H1("Weibull-Analyse Tool"),
        html.Br(id="test"),
        #html.Button("Herunterladen der Vorlage-Datei", id="btn"), dcc.Download(id="download"),

        #This doesnt work just YET
        #dbc.Button("Herunterladen der Vorlage-Datei", id="btn_link",href = "https://github.com/bil-hamburg/WeibullAnalysisTool/raw/main/Vorlage.csv"),
        html.Button("Herunterladen der Vorlage-Datei", id="btn"), dcc.Download(id="download"),
        html.H2("1. Hochladen der CSV-Datei"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Bitte die Datei durch heraufziehen oder durch Click hochladen',
                #html.A('Select Files')
            ]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                    "max-width": "500px"
                },
        ),
        dcc.Store(id='store'),

        #html.H2("2. Hochgeladene Datei"),

        #html.Ul(id="file-list"),
        html.H2("2. Tabelleninhalt (5 Zeilen) "),
        html.Div(id='table-content'),
        #dcc.Tab(id="table-content", style={"max-width": "500px"}),

        ############Analysis section##############
        html.H2("3. Analyse"),

        # Analysis Info Text
        html.Div([
            html.P("Die Analyse ist möglich, sobald eine Datei hochgeladen wurde", id='info-text'),
        ], style={'display': 'block'}),

        # Analysis Button
        #html.Div([
        #    html.Button("Analyse starten", id="analysis-button", style={'display': 'none'}),
        #]),
        dcc.Store(id="analysis-data", storage_type='session'),
        # Kennzahlen Datentabelle
        dcc.Tab(id="Kennzahlen-content", style={"max-width": "800px"}),
        # Plot
        html.Div(
            dcc.Graph(
                id='line_plot_zuverlässigkeit',
                # figure= dsf.customChart1(Data=Data, attribution=['Gender','JobType'], title='Gender Distribution'),
                style={'width': '800', 'display': 'none'}
            ), style={'display': 'inline-block'}),
        html.Div(
            dcc.Graph(
                id='line_plot_Ausfallrate',
                style={'width': '800', 'display': 'none'}
            ), style={'display': 'inline-block'}),
        html.Div(
            dcc.Graph(
                id='line_plot_Wahrscheinlichkeitsverteilungsfunktion',
                # figure= dsf.customChart1(Data=Data, attribution=['Gender','JobType'], title='Gender Distribution'),
                style={'width': '800', 'display': 'none'}
            ), style={'display': 'inline-block'}),
        html.Div(
            dcc.Graph(
                id='line_plot_KumulativeVerteillungsfunktion',
                style={'width': '800', 'display': 'none'}
            ), style={'display': 'inline-block'}),

        # Download Button
        # html.Div([
        html.Button("Dateien als Word-Datei herunterladen", id="download-button", style={'display': 'none'}),
        dcc.Download(id="download-report"),
        dcc.Store(id='Beta'),
        dcc.Store(id='Eta'),
        dcc.Store(id='MTTF'),

    ],

)


# # NEW
# @app.callback(Output('confirm-danger', 'displayed'),
#               Input('dropdown-danger', 'value'))
# def display_confirm(value):
#     if value == 'Danger!!':
#         return True
#     return False
#
#
# @app.callback(Output('output-danger', 'children'),
#               Input('confirm-danger', 'submit_n_clicks'))
# def update_output(submit_n_clicks):
#     if submit_n_clicks:
#         return 'It wasnt easy but we did it {}'.format(submit_n_clicks)


# Vorgehen:
# Callback der bei upload ausgeführt
# check auf Datenvalidität:
# Check negative Laufzeit
# valide eintragungen "Ausgefallen"...
# 2 Outputs
# Format pop up IF true
# else ausrechnen


# NEW


@app.callback(Output("download", "data"), [Input("btn", "n_clicks")], prevent_initial_call=True)
def Func(n_clicks):
    return dcc.send_data_frame(df_CSV.to_csv, "Vorlage.csv", index=True)


# In[4]:


# Auto hide on startup
# @app.callback(Output("analysis-button", "style"), [Input("test", "hidden")])
# def Func2(n_clicks):
#    print("test")
#    return {'display': 'none'}
# return dcc.send_file("Vorlage.xlsx")
@app.callback(Output('store', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(contents, list_of_names, list_of_dates):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    #print(content_string)
    decoded = base64.b64decode(content_string)
    #print(decoded)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';')
    #print(df)
    #df = pd.read_excel(io.StringIO(decoded.decode('utf-8')))
    return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('table-content', 'children'),
    Input('store', 'data')
)
def output_from_store(stored_data):
    df = pd.read_json(stored_data, orient='split')
    df=df.head(5)
    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            fill_width=False
        ),
        #html.Hr(),
    ])


# @app.callback(
#     Output("table-content", "children"),
#     [Input('store', 'data')], prevent_initial_call=True
# )
# def TableDisplay(stored_data):
#     data_split=stored_data.split(';')
#     pd_obj = pd.read_json(data_split, orient='split')
#
#     df=pd.DataFrame(pd_obj)
#
#     df_small = df.head(5)
#     print(df)
#
#     table_small = dash_table.DataTable(df_small.to_dict('records'), fill_width=False)
#     #table_small = dash_table.DataTable(df)
#
#     return table_small



def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    if os.path.exists(UPLOAD_DIRECTORY):
        # os.remove(UPLOAD_DIRECTORY)
        for filename in os.listdir(UPLOAD_DIRECTORY):
            # print(UPLOAD_DIRECTORY+"/"+filename)
            os.remove(UPLOAD_DIRECTORY + "/" + filename)

    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)

    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)



@app.callback(
    Output(component_id="info-text", component_property="style"),
    [Input(component_id="table-content", component_property="children")], prevent_initial_call=True
)
def hidetext(content):
    return {'display': 'none'}


# @app.callback(
#     Output(component_id="analysis-button", component_property="style"),
#     [Input(component_id="info-text", component_property="style")], prevent_initial_call=True
# )
# def showbutton(children):
#     return {'display': 'block'}


# Analysis Button
@app.callback([
    Output(component_id="line_plot_zuverlässigkeit", component_property="style"),
    Output("line_plot_zuverlässigkeit", "figure"),
    Output(component_id="line_plot_Ausfallrate", component_property="style"),
    Output("line_plot_Ausfallrate", "figure"),
    Output(component_id="line_plot_KumulativeVerteillungsfunktion", component_property="style"),
    Output("line_plot_KumulativeVerteillungsfunktion", "figure"),
    Output(component_id="line_plot_Wahrscheinlichkeitsverteilungsfunktion", component_property="style"),
    Output("line_plot_Wahrscheinlichkeitsverteilungsfunktion", "figure"),
    Output("Kennzahlen-content", "children"),
    Output(component_id="download-button", component_property="style"),
    Output('Beta', 'data'),
    Output('Eta', 'data'),
    Output('MTTF', 'data')
],
    # Output(component_id="analysis-button", component_property="style")],
    #[Input("analysis-button", "n_clicks"), Input("store","data")], prevent_initial_call=True
    [Input("store","data")], prevent_initial_call=True
)
def StartAnalysis(stored_data):
    # WeibullAPP
    #file = os.listdir(UPLOAD_DIRECTORY)
    #filename = UPLOAD_DIRECTORY + '/' + file[0]
    #RawData = pd.read_excel(filename)
    RawDataPD = pd.read_json(stored_data, orient='split')
    RawData=pd.DataFrame(RawDataPD)
    df = RawData.sort_values('Laufzeit')
    # Todo: data check(valid)+cleaning
    df.Rank = range(1, len(RawData.index) + 1)
    df.ReverseRank = range(len(RawData.index), 0, -1)
    df = df.reset_index(drop=True)
    df['AdjustedRank'] = 0
    df['Rank'] = range(1, len(RawData.index) + 1)
    df['ReverseRank'] = range(len(RawData.index), 0, -1)
    df['MedianRank'] = np.NaN
    for i in range(0, len(RawData.index)):
        # checking first entry
        if i == 0:
            if df.AusgefallenOderInBetrieb[0] == "In Betrieb":
                LastAdjustedRank = 0
                df.AdjustedRank[0] = "In Betrieb"
            else:
                df.AdjustedRank[0] = df.AdjustedRank[i] = (df.ReverseRank[i] * 0 + len(RawData.index) + 1) / (
                            df.ReverseRank[i] + 1)
                LastAdjustedRank = df.AdjustedRank[0]
                df.MedianRank[i] = (df.AdjustedRank[i] - 0.3) / (len(RawData.index) + 0.4)
        else:
            if df.AusgefallenOderInBetrieb[i] == "In Betrieb":
                df.AdjustedRank[i] = "In Betrieb"
            else:
                df.AdjustedRank[i] = (df.ReverseRank[i] * LastAdjustedRank + len(RawData.index) + 1) / (
                            df.ReverseRank[i] + 1)
                LastAdjustedRank = df.AdjustedRank[i]
                df.MedianRank[i] = (df.AdjustedRank[i] - 0.3) / (len(RawData.index) + 0.4)

    #print(df.head(5))
    LnT = []
    for i in range(0, len(RawData.index)):
        LnT.append(math.log(df.Laufzeit[i]))
    df['LnT'] = LnT
    lnlnMedianRank = []
    for i in range(0, len(RawData.index)):
        if not np.isnan(df.MedianRank[i]):
            # temp=math.log(1-df.MedianRank[i])
            # lnlnMedianRank.append(math.log(-temp))
            lnlnMedianRank.append(math.log(-math.log(1 - df.MedianRank[i])))
        else:
            lnlnMedianRank.append(np.nan)
    df['lnlnMedianRank'] = lnlnMedianRank
    #print(df.head(5))

    # Beta approximation through slope

    # converting for float isnan check
    def is_nan(value):
        return math.isnan(float(value))

    lnlnMedianRank = []
    LnT = []
    for i in range(0, len(RawData.index)):
        if not is_nan(df.lnlnMedianRank[i]):
            lnlnMedianRank.append(df.lnlnMedianRank[i])
            LnT.append(df.LnT[i])
    LnTA = np.array(LnT)
    lnlnMedianRankA = np.array(lnlnMedianRank)
    model = LinearRegression().fit(LnTA.reshape((-1, 1)), lnlnMedianRankA)
    print("Beta is: ", model.coef_[0])
    Beta = model.coef_[0]

    # Result: Compared to the Excel solution, this one differs by 0.01 in comparision due to slope approximation. Depending on regression model used.
    # sufficient
    # intercept and eta

    # intercept function

    def Intercept(R1, R2):
        sumbot = 0
        sumtop = 0
        for i in range(0, len(R1)):
            AverageR1 = np.mean(R1)
            AverageR2 = np.mean(R2)
            sumtop += (R1[i] - AverageR1) * (R2[i] - AverageR2)
            sumbot += pow((R1[i] - AverageR1), 2)
        b = sumtop / sumbot
        a = AverageR2 - b * AverageR1
        return a

    intercept = Intercept(LnTA, lnlnMedianRankA)
    # print(intercept)

    Eta = math.exp(-intercept / Beta)
    print("Eta is: ", Eta)
    # eta differs by 0,8, sufficient
    print("MTTF is : TO BE ANALYSED (durchschnittliche Lebenszeit)")
    MTTF = 0
    CharacteristicLifetime = Eta
    print("Characterstic Lifetime is : ", Eta, " 63.2% of Units fail to this point")
    df_Plots = pd.DataFrame()
    stepwidth = (df.Laufzeit.max() / len(df.Laufzeit)) / 10
    # print(stepwidth)
    # stepwidth=10
    df_Plots.insert(0, "Laufzeit", [0])
    df_Plots['Zuverlässigkeit'] = 1  # math.exp(-pow((df.Laufzeit[0]/Eta),Beta))
    df_Plots['Wahrscheinlichkeitsdichtefunktion'] = 0
    df_Plots['Ausfallrate'] = 0
    df_Plots['KumulativeVerteillungsfunktion'] = 0
    i = 1
    df_temp = pd.DataFrame()
    while df_Plots.Zuverlässigkeit[i - 1] > 0.005:
        Laufzeit_Temp = stepwidth + df_Plots.Laufzeit[i - 1]
        Zuverlässigkeit_Temp = math.exp(-pow((Laufzeit_Temp / Eta), Beta))
        Wahrscheinlichkeitsdichtefunktion_Temp = (Beta / Eta) * pow((Laufzeit_Temp / Eta), Beta - 1) * math.exp(
            -pow((Laufzeit_Temp / Eta), Beta))
        Ausfallrate_Temp = (Beta / Eta) * pow((Laufzeit_Temp / Eta), Beta - 1)
        KumulativeVerteillungsfunktion_Temp = 1 - Zuverlässigkeit_Temp
        new_Row = pd.Series(
            data={'Laufzeit': Laufzeit_Temp, 'Zuverlässigkeit': Zuverlässigkeit_Temp, 'Ausfallrate': Ausfallrate_Temp,
                  'KumulativeVerteillungsfunktion': KumulativeVerteillungsfunktion_Temp,
                  'Wahrscheinlichkeitsdichtefunktion': Wahrscheinlichkeitsdichtefunktion_Temp})
        #df_Plots = df_Plots._append(new_Row, ignore_index=True)
        df_Plots = df_Plots.append(new_Row, ignore_index=True)

        i += 1

    # add further plot parameters

    # plot
    df_Plots.plot(x='Laufzeit', y="Zuverlässigkeit")
    # figure_zuverlässigkeit = go.Figure(
    #    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    #    layout=go.Layout(
    #     title=go.layout.Title(text="A Figure Specified By A Graph Object")
    #     )
    # )
    figure_zuverlässigkeit = px.line(df_Plots, x="Laufzeit", y="Zuverlässigkeit",
                                     template="simple_white")
    figure_zuverlässigkeit.update_yaxes(showgrid=True)
    figure_zuverlässigkeit.update_xaxes(showgrid=True)
    # Datetimenow=str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    # Analysis_Directory=UPLOAD_DIRECTORY+'/'+Datetimenow+'/'
    # os.makedirs(Analysis_Directory)
    Analysis_Directory = UPLOAD_DIRECTORY
    # print(Analysis_Directory)
    figure_zuverlässigkeit.update_layout(
        title=dict(
            text='<b>Zuverlässigkeit</b>',
            x=0.5,
            # y=0.95,
            font=dict(
                family="Arial",
                size=20,
                color='#000000'
            )
        )
    )
    #figure_zuverlässigkeit.write_image(Analysis_Directory + '/Zuverlässigkeit.png')  # add date

    figure_Ausfallrate = px.line(df_Plots, x="Laufzeit", y="Ausfallrate", template="simple_white")
    figure_Ausfallrate.update_yaxes(showgrid=True)
    figure_Ausfallrate.update_xaxes(showgrid=True)
    figure_Ausfallrate.update_layout(
        title=dict(
            text='<b>Ausfallrate</b>',
            x=0.5,
            # y=0.95,
            font=dict(
                family="Arial",
                size=20,
                color='#000000'
            )
        )
    )
    figure_Ausfallrate.update_layout(
        xaxis_title="Laufzeit", yaxis_title="Ausfallrate"
    )

    #figure_Ausfallrate.write_image(Analysis_Directory + '/Ausfallrate.png')
    # P(x)
    figure_Wahrscheinlichkeitsverteilungsfunktion = px.line(df_Plots, x="Laufzeit",
                                                            y="Wahrscheinlichkeitsdichtefunktion",
                                                            template="simple_white")
    figure_Wahrscheinlichkeitsverteilungsfunktion.update_yaxes(showgrid=True)
    figure_Wahrscheinlichkeitsverteilungsfunktion.update_xaxes(showgrid=True)
    figure_Wahrscheinlichkeitsverteilungsfunktion.update_layout(
        title=dict(
            text='<b>Wahrscheinlichkeitsdichtefunktion</b>',
            x=0.5,
            # y=0.95,
            font=dict(
                family="Arial",
                size=20,
                color='#000000'
            )
        )
    )
    figure_Ausfallrate.update_layout(
        xaxis_title="Laufzeit", yaxis_title="Ausfallwahrscheinlichkeit"
    )
    # Kumulative Ausfälle (bezogen auf die Gesamtmenge)
    #figure_Wahrscheinlichkeitsverteilungsfunktion.write_image(
    #    Analysis_Directory + '/Wahrscheinlichkeitsverteilungsfunktion.png')

    figure_KumulativeVerteillungsfunktion = px.line(df_Plots, x="Laufzeit", y="KumulativeVerteillungsfunktion",
                                                    template="simple_white")
    figure_KumulativeVerteillungsfunktion.update_yaxes(showgrid=True)
    figure_KumulativeVerteillungsfunktion.update_xaxes(showgrid=True)
    figure_KumulativeVerteillungsfunktion.update_layout(
        title=dict(
            text='<b>Kumulative Verteillungsfunktion</b>',

            # yaxis_title='Kumulative Ausfälle (Bezogen auf die Gesamtmenge)',
            x=0.5,
            # y=0.95,
            font=dict(
                family="Arial",
                size=20,
                color='#000000'
            )
        )
    )
    figure_KumulativeVerteillungsfunktion.update_layout(
        xaxis_title="Laufzeit", yaxis_title="Kumulative Ausfälle (Bezogen auf die Gesamtmenge)"
    )
    # figure_KumulativeVerteillungsfunktion.update_yaxes(y = 'Kumulative Ausfälle (Bezogen auf die Gesamtmenge)')
    #figure_KumulativeVerteillungsfunktion.write_image(Analysis_Directory + '/KumulativeVerteillungsfunktion.png')

    df_Plots.plot(x='Laufzeit', y="Ausfallrate")
    df_Plots.plot(x='Laufzeit', y="KumulativeVerteillungsfunktion")
    df_Plots.plot(x='Laufzeit', y="Wahrscheinlichkeitsdichtefunktion")
    # End WeibullApp

    # Kennzahlen Tabelle
    # MORE TO BE DONE
    Kennzahlen_Data = {'Kennzahl': ['Beta', 'Charakteristische Lebensdauer(Eta)', 'MTTF'], 'Wert': [Beta, Eta, 'tbd']}
    df_Kennzahlen = pd.DataFrame(Kennzahlen_Data)
    # table_Kennzahlen=dash_table.DataTable(df_Kennzahlen.to_dict(), [{"Kennzahl": i, "Wert": i} for i in df.columns], fill_width=False)
    #table_Kennzahlen = dash_table.DataTable(df_Kennzahlen.to_dict('rows'), fill_width=False)
    table_Kennzahlen = dash_table.DataTable(df_Kennzahlen.to_dict(), fill_width=False)

    print(df_Kennzahlen)
    # SaveValues(Beta, Eta, MTTF):

    #    return {'display': 'block'},figure_zuverlässigkeit,{'display': 'block'}, figure_Ausfallrate,{'display': 'block'},figure_KumulativeVerteillungsfunktion,{'display': 'block'},figure_Wahrscheinlichkeitsverteilungsfunktion
    return {'display': 'block'}, figure_zuverlässigkeit, {'display': 'block'}, figure_Ausfallrate, {
        'display': 'block'}, figure_KumulativeVerteillungsfunktion, {
        'display': 'block'}, figure_Wahrscheinlichkeitsverteilungsfunktion, table_Kennzahlen, {
        'display': 'block'}, Beta, Eta, MTTF


@app.callback(Output("download-report", "data"), [Input("download-button", "n_clicks")], State('Beta', 'data'),
              State('Eta', 'data'), State('MTTF', 'data'), prevent_initial_call=True)
def CreateAndDownloadDocx(n_clicks, Beta, Eta, MTTF):
    template = DocxTemplate(
        'C:/Users\denni/OneDrive - haw-hamburg.de/BIL Arbeitsordner/03_Data driven business/WeibullAnalysisTool/WeibullReportVorlage.docx')
    Analysis_Directory = UPLOAD_DIRECTORY

    # import figures
    ImgZuverlässigkeit = InlineImage(template, Analysis_Directory + '/Zuverlässigkeit.png', Cm(15))
    ImgWahrscheinlichkeitsdichtefunktion = InlineImage(template,
                                                       Analysis_Directory + '/Wahrscheinlichkeitsverteilungsfunktion.png',
                                                       Cm(15))
    ImgAusfallrate = InlineImage(template, Analysis_Directory + '/Ausfallrate.png', Cm(15))
    ImgKumulativeVerteillungsfunktion = InlineImage(template,
                                                    Analysis_Directory + '/KumulativeVerteillungsfunktion.png', Cm(15))

    # template = DocxTemplate('WeibullReport.docx')
    context = {
        'title': 'Weibull-Analyse Report',
        'day': datetime.now().strftime('%d'),
        'month': datetime.now().strftime('%m'),
        'year': datetime.now().strftime('%Y'),
        # 'figure1': figure_zuverlässigkeit
        'figure1': ImgZuverlässigkeit,
        'figure2': ImgAusfallrate,
        'figure3': ImgWahrscheinlichkeitsdichtefunktion,
        'figure4': ImgKumulativeVerteillungsfunktion,
        'beta_var': Beta,
        'eta_var': Eta,
        'mttf_var': "TBD"

    }

    template.render(context)
    template.save(Analysis_Directory + '/WeibullReport.docx')

    time.sleep(2)
    # plotly.offline.plot(figure_zuverlässigkeit,filename=(UPLOAD_DIRECTORY+"/Zuverlässigkeit.png')#date
    # figure_zuverlässigkeit.write_image(UPLOAD_DIRECTORY+'/Zuverlässigkeit.png')

    return dcc.send_file(Analysis_Directory + '/WeibullReport.docx')


#    return dcc.send_file("Vorlage.xlsx")

# return dcc.send_file(Analysis_Directory+'/WeibullReport.docx')#temp


# Format error file
# Plot headline (Titel "Zuverlässigkeit"...)
# Download Word
# Word: add inserted datatable
# save file function: Do not delete data


# In[5]:


if __name__ == '__main__':
    app.run_server(port=8002, debug=False)
    # app.run(debug=True, port=8001)




