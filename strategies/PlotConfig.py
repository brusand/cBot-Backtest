from strategies.Indicators import *;

class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                resample('bollinger_upperband') : {'color': 'rgba(4,137,122,0.7)'},
                resample('kc_upperband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_middleband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_lowerband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('bollinger_lowerband') : {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': resample('bollinger_upperband'),
                    'fill_color': 'rgba(4,137,122,0.07)'
                    },
                resample('ema9') : {'color': 'purple'},
                resample('ema20') : {'color': 'yellow'},
                resample('ema50') : {'color': 'red'},
                resample('ema200') : {'color': 'white'},
            },
            'subplots': {
                "ATR" : {
                    resample('atr'):{'color':'firebrick'}
                },
                "ADX": {
                    resample('adx'): {'color': 'firebrick'}
                },
                "MFI": {
                    resample('mfi'): {'color': 'firebrick'}
                }
            }
        }

    def add_pivots_in_config(self):
        self.config['main_plot']["pivot_lows"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'olive'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        return self

    def add_divergence_in_config(self, indicator:str):
        # self.config['main_plot']["bullish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'orange'
        #         }
        #     }
        # }
        # self.config['main_plot']["bearish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'purple'
        #         }
        #     }
        # }
        for i in range(3):
            self.config['main_plot']["bullish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line' : {
                        'color': 'green',
                        'dash' :'dash'
                    }
                }
            }   
            self.config['main_plot']["bearish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line' : {
                        "color":'crimson',
                        'dash' :'dash'
                    }
                }
            } 
        return self

    def add_total_divergences_in_config(self, dataframe):
        total_bullish_divergences_count = dataframe[resample("total_bullish_divergences_count")]
        total_bullish_divergences_names = dataframe[resample("total_bullish_divergences_names")]
        self.config['main_plot'][resample("total_bullish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bullish_divergences_count,
                'hovertext': total_bullish_divergences_names,
                'textfont':{'size': 11, 'color':'green'},
                'textposition':'bottom center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'green'
                }
            }
        }
        total_bearish_divergences_count = dataframe[resample("total_bearish_divergences_count")]
        total_bearish_divergences_names = dataframe[resample("total_bearish_divergences_names")]
        self.config['main_plot'][resample("total_bearish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bearish_divergences_count,
                'hovertext': total_bearish_divergences_names,
                'textfont':{'size': 11, 'color':'crimson'},
                'textposition':'top center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'crimson'
                }
            }
        }
        return self