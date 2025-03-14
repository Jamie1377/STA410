import yfinance as yf
from stock_prediction.core.hmm_model import full_process_hmm

strong_buy_fin_companies = yf.Sector("financial-services").top_companies[
    yf.Sector("financial-services").top_companies["rating"] == "Buy"
]
big_fin = strong_buy_fin_companies["market weight"] > 0.01
big_fin_companies = strong_buy_fin_companies[big_fin]
big_fin_companies_list = list(big_fin_companies.index)
strong_buy_companies = yf.Sector("technology").top_companies[
    yf.Sector("technology").top_companies["rating"] == "Strong Buy"
]
strong_buy_list = list(strong_buy_companies.index)[0:2]


full_process_hmm(big_fin_companies_list[:5], "2024-01-01", NUM_ITERS=250)
