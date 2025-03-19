import warnings
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import yfinance as yf
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)
from stock_prediction.utils import get_next_valid_date
import pandas_market_calendars as mcal
from datetime import timedelta

PLOT_SHOW = True
PLOT_TYPE = False

NUM_TEST = 100
K = 80
NUM_ITERS = 10000
NUM_FUTURE_DAYS = 10
STOCKS = [
    "AAPL",
    "CRWD",
    "AVGO",
]
labels = ["Open", "High", "Low", "Close"]
likelihood_vect = np.empty([0, 1])
aic_vect = np.empty([0, 1])
bic_vect = np.empty([0, 1])

# Possible number of states in Markov Model
STATE_SPACE = range(2, 15)


# Calculating Mean Absolute Percentage Error of predictions
def calc_mape(predicted_data, true_data):
    return np.divide(
        np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0),
        true_data.shape[0],
    )


def get_trading_dates(start, end):
    """Get NYSE trading dates between start and end"""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return schedule.index.date


def normalize_transmat(transmat):
    transmat = np.array(transmat, dtype=np.float64)  # Ensure float type
    row_sums = transmat.sum(axis=1, keepdims=True)  # Compute row sums

    # Avoid division by zero (if a row is all zeros, assign uniform probability)
    row_sums[row_sums == 0] = 1.0

    return transmat / row_sums  # Normalize each row to sum to 1


def initialize_transmat(n_states, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)  # For reproducibility

    transmat = np.random.rand(n_states, n_states)  # Random values
    return normalize_transmat(transmat)  # Normalize to sum to 1


def fix_zero_rows(transmat, epsilon=1e-6):
    """Ensure no row in the transition matrix is entirely zero."""
    transmat = np.array(transmat, dtype=np.float64)

    row_sums = transmat.sum(axis=1)
    zero_rows = row_sums == 0  # Identify zero rows

    if np.any(zero_rows):
        transmat[zero_rows] = epsilon  # Replace zero rows with small probabilities
        transmat /= transmat.sum(axis=1, keepdims=True)  # Re-normalize

    return transmat


transmat = initialize_transmat(5)


def validate_transmat(transmat):
    transmat = np.array(transmat)
    row_sums = transmat.sum(axis=1)

    if np.any(row_sums == 0):
        raise ValueError("Error: At least one row in transmat is all zeros.")
    if not np.allclose(row_sums, 1):
        raise ValueError(f"Error: Rows must sum to 1, got {row_sums}")


# Example:
transmat = np.array([[0.5, 0.5], [0.3, 0.7]])
validate_transmat(transmat)  # No error if valid


def fix_startprob(startprob):
    """Fix NaNs in startprob_ and normalize"""
    # Replace NaNs with a small value (e.g., 1e-6)
    startprob = np.nan_to_num(startprob, nan=1e-10)
    prob_sum = startprob.sum(keepdims=True)
    if prob_sum == 0:
        startprob = np.array([1] * len(startprob)) / (len(startprob))
    else:
        # # Normalize to sum to 1
        startprob /= prob_sum
    return startprob


def fix_transmat(transmat):
    """Normalize each row of transmat_ to sum to 1, handling NaN and zero rows properly."""
    transmat = np.array(transmat, dtype=np.float64)

    # Replace NaN values with a small positive number
    transmat = np.nan_to_num(transmat, nan=1e-10)

    # Compute row sums
    row_sums = transmat.sum(axis=1, keepdims=True)

    # Identify rows that sum to zero
    zero_rows = (row_sums == 0).flatten()

    # Replace zero rows with a uniform distribution
    transmat[zero_rows] = 1.0 / transmat.shape[1]

    # Recompute row sums and normalize each row
    row_sums = transmat.sum(axis=1, keepdims=True)
    transmat /= row_sums

    return transmat


import numpy as np
from hmmlearn.hmm import GaussianHMM


def train_hmm(dataset, n_components=4, n_iter=1000):
    """Train an HMM model with regularization and return the trained model"""
    X = np.column_stack([dataset.pct_change().dropna().values])  # Use log returns
    model = GaussianHMM(
        n_components=n_components, covariance_type="diag", n_iter=n_iter, tol=0.0001
    )
    model.fit(X)

    # Smooth transition matrix to avoid overconfidence in past states
    model.transmat_ = (
        0.9 * model.transmat_ + 0.1 * np.ones_like(model.transmat_) / model.n_components
    )
    return model


def forecast_hmm(model, dataset, steps=5):
    """Forecast next steps iteratively using HMM"""
    predictions = []
    past_likelihoods = []
    current_state = model.predict(dataset.pct_change().dropna().values.reshape(-1, 1))[
        -1
    ]

    for _ in range(steps):
        next_obs = model.sample(1)[0].flatten()[0]  # Sample from HMM distribution
        past_likelihoods.append(
            model.score(dataset.pct_change().dropna().values.reshape(-1, 1))
        )

        # Apply correction to prevent drift
        correction_factor = (
            np.mean(past_likelihoods[-5:]) if len(past_likelihoods) >= 5 else 0
        )
        next_obs_corrected = (
            next_obs + correction_factor * 0.05
        )  # Small correction to mitigate drift

        # Convert log returns back to price
        last_price = dataset.iloc[-1]
        predicted_price = last_price * (1 + next_obs_corrected)
        predictions.append(predicted_price)

        # Update dataset iteratively
        dataset = pd.merge(dataset, pd.Series(predicted_price))
        # dataset = pd.

    return predictions


from datetime import date


def full_process_hmm(STOCKS=["AVGO"], start="2020-01-01", NUM_ITERS=10000):
    # np.random.seed(420)

    # start = '2023-04-01'
    for stock in STOCKS:
        dataset = yf.download(stock, start=start, end=date.today()).iloc[
            :-NUM_FUTURE_DAYS
        ]
        columns = dataset.columns
        if len(dataset) == 0:
            continue
        dataset_actual = yf.download(stock, start=start, end=date.today())
        prices = dataset["Close"].values.reshape(-1, 1)
        prices_actual = dataset_actual["Close"].values.reshape(-1, 1)
        predicted_stock_data = np.empty([0, dataset.shape[1]])
        likelihood_vect = np.empty([0, 1])
        aic_vect = np.empty([0, 1])
        bic_vect = np.empty([0, 1])
        for states in STATE_SPACE:
            num_params = states**2 + states
            dirichlet_params_states = np.random.randint(1, 50, states)
            # model = hmm.GaussianHMM(n_components=states, covariance_type='full', startprob_prior=dirichlet_params_states, transmat_prior=dirichlet_params_states, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
            model = hmm.GaussianHMM(
                n_components=states,
                covariance_type="diag",
                tol=0.0001,
                n_iter=NUM_ITERS,
            )
            
            model.fit(dataset.iloc[NUM_TEST:, :])
            if not model.monitor_.converged:
                print("Model did not converge, skipping...")
            else:
                print("Model converged successfully!")



            
            model.startprob_ = fix_startprob(model.startprob_)
            model.transmat_ = fix_transmat(model.transmat_)
            # if model.monitor_.iter == NUM_ITERS:
            #     print('Increase number of iterations')
            #     sys.exit(1)
            likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
            aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
            bic_vect = np.vstack(
                (
                    bic_vect,
                    -2 * model.score(dataset) + num_params * np.log(dataset.shape[0]),
                )
            )

        opt_states = np.argmin(bic_vect) + 2
        print("Optimum number of states are {}".format(opt_states))

        for idx in reversed(range(NUM_TEST)):
            train_dataset = dataset.iloc[idx + 1 :, :]
            test_data = dataset.iloc[idx, :]
            num_examples = train_dataset.shape[0]
            # model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', startprob_prior=dirichlet_params, transmat_prior=dirichlet_params, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
            # if idx == NUM_TEST - 1:
            #     model = hmm.GaussianHMM(n_components=opt_states, covariance_type='diag', tol=0.0001, n_iter=NUM_ITERS, init_params='stmc')
            # else:
            #     # Retune the model by using the HMM paramters from the previous iterations as the prior
            #     # model.transmat_ = fix_zero_rows(transmat_retune_prior)
            #     transmat_retune_prior = fix_transmat(transmat_retune_prior)

            #     # model.startprob_ = startprob_retune_prior
            #     # model.means_ = means_retune_prior
            #     # model.covars_ = covars_retune_prior
            #     model = hmm.GaussianHMM(n_components=opt_states, covariance_type='diag', tol=0.0001, n_iter=NUM_ITERS, init_params='')

            #     model.transmat_ = transmat_retune_prior
            #     # model.startprob_ = np.maximum(startprob_retune_prior, 1e-6)  # Avoid zeros
            #     # model.startprob_ /= model.startprob_.sum()  # Normalize

            #     # Ensure means and covariances are valid before assigning
            #     # if means_retune_prior is not None:
            #     #     model.means_ = means_retune_prior
            #     # if covars_retune_prior is not None:
            #     #     model.covars_ = covars_retune_prior

            #     # model.transmat_ =  model.transmat_
            #     # model.startprob_ =  model.startprob_
            #     # model.means_ = model.means_
            #     # model.covars_ =  model.covars_
            model = model = hmm.GaussianHMM(
                n_components=opt_states,
                covariance_type="diag",
                tol=0.0001,
                n_iter=NUM_ITERS,
                init_params="stmc",
            )

            model.fit(np.flipud(train_dataset))
            model.transmat_ = fix_transmat(model.transmat_)
            model.startprob_ = fix_startprob(model.startprob_)

            if not model.monitor_.converged:
                print("Model did not converge, skipping...")
                continue
            else:
                print("Model converged successfully!")


            # model.transmat_ = transmat_retune_prior
            # transmat_retune_prior = model.transmat_
            # startprob_retune_prior = model.startprob_
            # means_retune_prior = model.means_
            # covars_retune_prior = model.covars_

            # if model.monitor_.iter == NUM_ITERS:
            #     print('Increase number of iterations')
            #     sys.exit(1)
            # print('Model score : ', model.score(dataset))
            # print('Dirichlet parameters : ',dirichlet_params)

            iters = 1
            past_likelihood = []
            curr_likelihood = model.score(np.flipud(train_dataset.iloc[0 : K - 1, :]))
            while iters < num_examples / K - 1:
                past_likelihood = np.append(
                    past_likelihood,
                    model.score(
                        np.flipud(train_dataset.iloc[iters : iters + K - 1, :])
                    ),
                )
                iters = iters + 1
            likelihood_diff_idx = np.argmin(
                np.absolute(np.array(past_likelihood) - curr_likelihood)
            )
            predicted_change = (
                train_dataset.iloc[likelihood_diff_idx, :]
                - train_dataset.iloc[likelihood_diff_idx + 1, :]
            )
            predicted_stock_data = np.vstack(
                (predicted_stock_data, dataset.iloc[idx + 1, :] + predicted_change)
            )
        # np.savetxt(
        #     "{}_forecast.csv".format(stock),
        #     predicted_stock_data,
        #     delimiter=",",
        #     fmt="%.2f",
        # )

        if PLOT_TYPE:
            hdl_p = plt.plot(range(100), predicted_stock_data)
            plt.title("Predicted stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
            plt.figure()
            hdl_a = plt.plot(range(100), np.flipud(dataset.iloc[range(100), :]))
            plt.title("Actual stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
        else:
            for i in range(4):
                plt.figure(figsize=(12, 8))
                plt.plot(
                    range(100),
                    predicted_stock_data[:, i],
                    "k-",
                    label="Predicted " + labels[i] + " price",
                )
                plt.plot(
                    range(100),
                    np.flipud(dataset.iloc[range(100), i]),
                    "r--",
                    label="Actual " + labels[i] + " price",
                )
                plt.xlabel("Time steps")
                plt.ylabel("Price")
                plt.title(labels[i] + " price" + " for " + stock[:-4] + f"{stock}")
                plt.grid(True)
                plt.legend(loc="upper left")

        print("-" * 100)

        predicted_stock_data = np.empty([0, dataset.shape[1]])

        # Iterate over the last NUM_TEST rows (reversed)
        for idx in range(len(dataset) - NUM_TEST, len(dataset)):
            # Training data: everything BEFORE the test index
            train_dataset = dataset.iloc[:idx, :]
            # Test data: current row (last NUM_TEST rows)
            test_data = dataset.iloc[idx, :]
            num_examples = train_dataset.shape[0]
            # Initialize model
            if idx == len(dataset) - 1:  # First iteration (most recent data)
                model = hmm.GaussianHMM(
                    n_components=opt_states,  # Replace with your optimal states
                    covariance_type="diag",
                    tol=0.0001,
                    n_iter=NUM_ITERS,
                    init_params="stmc",
                )
            else:
                # Use parameters from previous model (uncomment to activate)
                model = hmm.GaussianHMM(
                    n_components=3,
                    covariance_type="diag",
                    tol=0.0001,
                    n_iter=NUM_ITERS,
                    init_params="",
                )
                # model.transmat_ = transmat_retune_prior
                # model.startprob_ = startprob_retune_prior
                # model.means_ = means_retune_prior
                # model.covars_ = covars_retune_prior

            # Train on historical data (no flipud)
            model.fit(np.flipud(train_dataset))
            model.transmat_ = fix_transmat(model.transmat_)
            model.startprob_ = fix_startprob(model.startprob_)

            # # Check convergence
            # if model.monitor_.iter == NUM_ITERS:
            #     print('Increase number of iterations')
            #     sys.exit(1)

            # Likelihood-based prediction logic
            past_likelihood = []
            curr_likelihood = model.score(
                np.flipud(train_dataset.iloc[-K:, :])
            )  # Use recent K points

            # # Compare with historical windows
            # for window_start in range(0, len(train_dataset) - K):
            #     window = train_dataset.iloc[window_start:window_start + K, :]
            #     past_likelihood.append(model.score(window))
            #     # np.append(past_likelihood, model.score(window))

            iters = 1
            past_likelihood = []
            # curr_likelihood = model.score(np.flipud(train_dataset.iloc[0:K - 1, :]))
            curr_likelihood = model.score(
                train_dataset.iloc[-K:, :]
            )  # Use recent K points
            while iters < num_examples / K - 1:
                # past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset.iloc[iters:iters + K - 1, :])))
                past_likelihood.append(
                    model.score(np.flipud(train_dataset.iloc[iters : iters + K - 1, :]))
                )
                iters = iters + 1
            #     likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            #     predicted_change = train_dataset.iloc[likelihood_diff_idx,:] - train_dataset.iloc[likelihood_diff_idx + 1,:]
            #     predicted_stock_data = np.vstack((predicted_stock_data, dataset.iloc[idx + 1,:] + predicted_change))
            # np.savetxt('{}_forecast.csv'.format(stock),predicted_stock_data,delimiter=',',fmt='%.2f')

            # Find most similar historical window
            likelihood_diff_idx = np.argmin(
                np.absolute(np.array(past_likelihood) - curr_likelihood)
            )
            predicted_change = (
                train_dataset.iloc[likelihood_diff_idx + K, :]
                - train_dataset.iloc[likelihood_diff_idx + K - 1, :]
            )

            # Update predictions
            predicted_stock_data = np.vstack(
                (predicted_stock_data, dataset.iloc[idx, :] + predicted_change)
            )

            # # Update predictions (corrected)
            # predicted_value = train_dataset.iloc[-1, :] + predicted_change  # Predict next step from last training point
            # predicted_stock_data = np.vstack((predicted_stock_data, predicted_value))

        # Save forecasts
        # np.savetxt(
        #     f"{stock}_forecast.csv", predicted_stock_data, delimiter=",", fmt="%.2f"
        # )

        # mape = calc_mape(predicted_stock_data, np.flipud(dataset.iloc[range(100),:]))
        # print('MAPE for the stock {} is '.format(stock),mape)
        # model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
        # model.fit(prices[:-NUM_FUTURE_DAYS])
        # predicted_stock_data = model.predict(prices[:-NUM_FUTURE_DAYS])
        predicted_stock_data = pd.DataFrame(
            predicted_stock_data, columns=dataset.columns
        )

        if PLOT_TYPE:
            hdl_p = plt.plot(range(100), predicted_stock_data)
            plt.title("Predicted stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
            plt.figure()
            hdl_a = plt.plot(range(100), np.flipud(dataset.iloc[range(100), :]))
            plt.title("Actual stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
        else:
            for i in range(4):
                plt.figure(figsize=(12, 8))
                plt.plot(
                    (dataset.iloc[len(dataset) - NUM_TEST : len(dataset), i]).index,
                    predicted_stock_data.iloc[:, i],
                    "k-",
                    label="Predicted " + labels[i] + " price",
                )
                plt.plot(
                    (dataset.iloc[len(dataset) - NUM_TEST : len(dataset), i]).index,
                    (dataset.iloc[len(dataset) - NUM_TEST : len(dataset), i]),
                    "r--",
                    label="Actual " + labels[i] + " price",
                )
                plt.xlabel("Time steps")
                plt.ylabel("Price")
                plt.title(labels[i] + " price" + " for " + stock[:-4] + f"{stock}")
                plt.grid(True)
                plt.legend(loc="upper left")
        print("-" * 100)

        # if PLOT_SHOW:
        #     plt.show(block=False)

        # Generate future dates
        last_date = dataset.index[-1]
        future_dates = get_trading_dates(
            last_date + timedelta(days=1),
            last_date + timedelta(days=NUM_FUTURE_DAYS * 2),
        )
        future_dates = future_dates[:NUM_FUTURE_DAYS]

        last_date_act = dataset_actual.index[-1]
        future_dates_act = get_trading_dates(
            last_date_act + timedelta(days=1),
            last_date_act + timedelta(days=NUM_FUTURE_DAYS * 2),
        )
        future_dates_act = future_dates_act[:NUM_FUTURE_DAYS]

        predicted_stock_data = np.empty([0, dataset.shape[1]])
        predicted_stock_data_1 = dataset.iloc[-1].to_frame().T
        predicted_stock_data_1 = np.empty([0, dataset.shape[1]])

        predicted_stock_data_act = np.empty([0, dataset.shape[1]])
        predicted_stock_data_w = np.empty([0, dataset.shape[1]])
        # np.concatenate(np.empty([0, dataset_actual.shape[1]]))

        # actual_dataset = dataset.iloc[-1].to_frame().T
        actual_dataset = dataset.iloc[-(K + 1) :, :].copy()
        # wild_dataset = dataset_actual.iloc[-1].to_frame().T
        wild_dataset = dataset_actual.iloc[-(K + 1) :, :].copy()
        for idx in range(NUM_FUTURE_DAYS):

            num_examples = dataset.shape[0]

            if idx == len(dataset) - 1:  # First iteration (most recent data)
                model = hmm.GaussianHMM(
                    n_components=opt_states,  # Replace with your optimal states
                    covariance_type="diag",
                    tol=0.0001,
                    n_iter=NUM_ITERS,
                    init_params="stmc",
                )
            else:
                # Use parameters from previous model (uncomment to activate)
                model = hmm.GaussianHMM(
                    n_components=opt_states,
                    covariance_type="diag",
                    tol=0.0001,
                    n_iter=NUM_ITERS,
                    init_params="",
                )
                # model.transmat_ = transmat_retune_prior
                # model.startprob_ = startprob_retune_prior
                # model.means_ = means_retune_prior
                # model.covars_ = covars_retune_prior

            # Train on historical data (no flipud)
            model.fit(np.flipud(dataset))
            model.transmat_ = fix_transmat(model.transmat_)
            model.startprob_ = fix_startprob(model.startprob_)

            final_model = hmm.GaussianHMM(
                n_components=opt_states,  # Replace with your optimal states
                covariance_type="diag",
                tol=0.0001,
                n_iter=NUM_ITERS,
                init_params="stmc",
            )
            final_model.fit(np.flipud(dataset_actual))
            final_model.transmat_ = fix_transmat(final_model.transmat_)
            final_model.startprob_ = fix_startprob(final_model.startprob_)
            if not model.monitor_.converged:
                print("Model did not converge, skipping...")
                continue
            else:
                print("Model converged successfully!")

            # Check convergence
            # if model.monitor_.iter == NUM_ITERS:
            #     print('Increase number of iterations')
            #     sys.exit(1)

            # Likelihood-based prediction logic
            past_likelihood = []
            curr_likelihood = model.score(
                np.flipud(dataset.iloc[-K:, :])
            )  # Use recent K points

            past_likelihood_act = []
            curr_likelihood_act = final_model.score(
                np.flipud(dataset_actual.iloc[-K:, :])
            )  # Use recent K points

            iters = 1
            past_likelihood = []
            past_likelihood_act = []
            past_likelihood_w = []
            # curr_likelihood = model.score(np.flipud(train_dataset.iloc[0:K - 1, :]))
            curr_likelihood = model.score(dataset.iloc[-K:, :])  # Use recent K points
            curr_likelihood_act = final_model.score(dataset_actual.iloc[-K:, :])
            curr_likelihood_w = model.score(dataset_actual.iloc[-K:, :])

            while iters < num_examples / K - 1:
                # past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset.iloc[iters:iters + K - 1, :])))
                past_likelihood.append(
                    model.score(np.flipud(dataset.iloc[iters : iters + K - 1, :]))
                )
                iters = iters + 1

            likelihood_diff_idx = np.argmin(
                np.absolute(np.array(past_likelihood) - curr_likelihood)
            )
            if likelihood_diff_idx + K >= actual_dataset.shape[0]:
                likelihood_diff_idx = actual_dataset.shape[0] - K - 1
            predicted_change = (
                actual_dataset.iloc[likelihood_diff_idx + K, :]
                - actual_dataset.iloc[likelihood_diff_idx + K - 1, :]
            )

            # # Update predictions
            # predicted_stock_data_1 = np.vstack((
            #     predicted_stock_data_1,
            #     # dataset_actual.iloc[len(dataset_actual) - NUM_FUTURE_DAYS + idx, :] + predicted_change
            #     dataset_actual.iloc[len(dataset_actual) - NUM_FUTURE_DAYS + idx, :] + predicted_change
            # ))

            # Update predictions
            predicted_stock_data_1 = np.vstack(
                (predicted_stock_data_1, actual_dataset.iloc[-1, :] + predicted_change)
            )

            actual_dataset = np.vstack(
                (actual_dataset, actual_dataset.iloc[-1, :] + predicted_change)
            )

            actual_dataset = pd.DataFrame(actual_dataset, columns=columns)

            # num_examples_act = dataset_actual.shape[0]
            # iters = 1
            # while iters < num_examples_act / K - 1:
            #     # past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset.iloc[iters:iters + K - 1, :])))
            #     past_likelihood_act.append(final_model.score(np.flipud(dataset_actual.iloc[iters:iters + K - 1, :])))
            #     iters = iters + 1

            # likelihood_diff_idx_act = np.argmin(np.absolute(np.array(past_likelihood_act) - curr_likelihood_act))
            # predicted_change_act = (
            #     dataset_actual.iloc[likelihood_diff_idx_act + K, :] -
            #     dataset_actual.iloc[likelihood_diff_idx_act + K - 1, :]
            # )

            # # Update predictions
            # predicted_stock_data_act = np.vstack((
            #     predicted_stock_data_act,
            #     dataset.iloc[-1 + idx, :] + predicted_change_act
            # ))

            # actual_dataset =  np.vstack((
            #     actual_dataset,
            #     actual_dataset.iloc[-1 + idx, :] + predicted_change_act
            # ))
            # actual_dataset = pd.DataFrame(actual_dataset, columns= columns)

            num_examples_w = dataset_actual.shape[0]
            iters = 1
            # while iters < num_examples_w / K - 1:
            while iters < num_examples_w / K - 1:

                # past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset.iloc[iters:iters + K - 1, :])))
                past_likelihood_w.append(
                    final_model.score(
                        np.flipud(dataset_actual.iloc[iters : iters + K - 1, :])
                    )
                )
                iters = iters + 1

            likelihood_diff_idx_w = np.argmin(
                np.absolute(np.array(past_likelihood_w) - curr_likelihood_w)
            )
            if likelihood_diff_idx_w + K >= wild_dataset.shape[0]:
                likelihood_diff_idx_w = wild_dataset.shape[0] - K - 1
            predicted_change_w = (
                wild_dataset.iloc[likelihood_diff_idx_w + K, :]
                - wild_dataset.iloc[likelihood_diff_idx_w + K - 1, :]
            )

            # Update predictions
            # predicted_stock_data_w = np.vstack((
            #     predicted_stock_data_w,
            #     wild_dataset.iloc[-1, :] + predicted_change_w
            # ))

            # wild_dataset =  np.vstack((
            #     wild_dataset,
            #     wild_dataset.iloc[-1, :] + predicted_change_w
            # ))

            # Update predictions with using previous data (not fully interative)
            predicted_stock_data_w = np.vstack(
                (
                    predicted_stock_data_w,
                    0.25 * wild_dataset.iloc[NUM_FUTURE_DAYS - idx, :]
                    + 0.75 * wild_dataset.iloc[-1, :]
                    + predicted_change_w,
                )
            )

            wild_dataset = np.vstack(
                (
                    wild_dataset,
                    0.25 * wild_dataset.iloc[NUM_FUTURE_DAYS - idx, :]
                    + 0.75 * wild_dataset.iloc[-1, :]
                    + predicted_change_w,
                )
            )

            wild_dataset = pd.DataFrame(wild_dataset, columns=columns)

        # Save forecasts
        # np.savetxt(
        #     f"{stock}_forecast.csv", predicted_stock_data_act, delimiter=",", fmt="%.2f"
        # )
        predicted_stock_data_1 = pd.DataFrame(
            predicted_stock_data_1, columns=dataset.columns
        )
        predicted_stock_data_act = pd.DataFrame(
            predicted_stock_data_act, columns=dataset_actual.columns
        )
        predicted_stock_data_w = pd.DataFrame(
            predicted_stock_data_w, columns=dataset_actual.columns
        )
        if PLOT_TYPE:
            hdl_p = plt.plot(range(100), predicted_stock_data_1)
            plt.title("Predicted stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
            plt.figure()
            hdl_a = plt.plot(range(100), np.flipud(dataset.iloc[range(100), :]))
            plt.title("Actual stock prices")
            plt.legend(iter(hdl_p), ("Close", "Open", "High", "Low"))
            plt.xlabel("Time steps")
            plt.ylabel("Price")
        else:
            for i in range(4):
                plt.figure(figsize=(12, 8))
                plt.plot(
                    future_dates,
                    predicted_stock_data_1.iloc[:, i],
                    "k-",
                    label="Predicted " + labels[i] + " price",
                )
                # plt.plot(future_dates, predicted_stock_data.iloc[:,i],'k-', label = 'Predicted '+labels[i]+' price');
                # plt.plot(future_dates_act, predicted_stock_data_act.iloc[:,i],'k-', label = 'Predicted '+labels[i]+' price (WILD!!!)');
                plt.plot(
                    future_dates_act,
                    predicted_stock_data_w.iloc[:, i],
                    "k-",
                    label="Predicted " + labels[i] + " price (WILD!!!)",
                )
                plt.plot(
                    future_dates,
                    (
                        dataset_actual.iloc[
                            len(dataset_actual) - NUM_FUTURE_DAYS : len(dataset_actual),
                            i,
                        ]
                    ),
                    "r--",
                    label="Actual " + labels[i] + " price",
                )

                plt.xlabel("Time steps")
                plt.ylabel("Price")
                plt.title(labels[i] + " price" + " for " + stock[:-4] + f"{stock}")
                plt.grid(True)
                plt.legend(loc="upper left")
                plt.show()

        # Train final model with optimal states
        # final_model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', n_iter=NUM_ITERS)
        # # final_model.fit(prices)

        # # Predict future hidden states
        # future_states = model.predict(prices[-NUM_FUTURE_DAYS:])  # Use recent data for prediction

        # Forecast future prices
        # last_price = prices[-1][0]
        # future_prices = [last_price]
        # for state in future_states:
        #     # Simulate price change based on state (example: random walk)
        #     price_change = np.random.normal(loc=0, scale=0.01)  # Adjust scale for volatility
        #     future_prices.append(future_prices[-1] * (1 + price_change))

        # # Create future dates

        # # future_dates = pd.date_range(start=dataset.index[-1], periods=NUM_FUTURE_DAYS + 1, freq='B')

        # today = dataset.index[-1]
        # future_dates = [today]
        # for _ in range(NUM_FUTURE_DAYS):
        #     nextday = get_next_valid_date(today)
        #     future_dates.append(nextday)
        #     today = nextday

        # future_dates = pd.DatetimeIndex(np.array(future_dates))

        # Combine historical and future data
        # historical_prices = dataset['Close']
        # forecast_prices = pd.Series(future_prices, index=future_dates)

        # # Plot results
        # plt.figure(figsize=(12, 6))
        # plt.plot(historical_prices.index, historical_prices, label='Historical Prices', color='blue')
        # plt.plot(forecast_prices.index, forecast_prices, label='Forecast Prices', color='red', linestyle='--')

        # plt.plot(dataset_actual.index, dataset_actual['Close'])
        # plt.title(f'{stock} Price Forecast')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # final_model.fit(prices_actual)

        # # Predict future hidden states
        # future_states_actual = final_model.predict(prices_actual[-NUM_FUTURE_DAYS:])  # Use recent data for prediction

        # # Forecast future prices
        # last_price_actual = prices_actual[-1][0]
        # future_prices_actual = [last_price_actual]
        # for state in future_states_actual:
        #     # Simulate price change based on state (example: random walk)
        #     price_change_actual = np.random.normal(loc=0, scale=0.01)  # Adjust scale for volatility
        #     future_prices_actual.append(future_prices_actual[-1] * (1 + price_change_actual))

        # # Create future dates

        # # future_dates_actual = pd.date_range(start=dataset_actual.index[-1], periods=NUM_FUTURE_DAYS + 1, freq='B')

        # today = dataset_actual.index[-1]
        # future_dates_actual = [today]
        # for _ in range(NUM_FUTURE_DAYS):
        #     nextday = get_next_valid_date(today)
        #     future_dates_actual.append(nextday)
        #     today = nextday

        # future_dates_actual = pd.DatetimeIndex(np.array(future_dates_actual))

        # # Combine historical and future data
        # historical_prices_actual = dataset_actual['Close']
        # forecast_prices_actual = pd.Series(future_prices_actual, index=future_dates_actual)

        # # Plot results
        # # plt.figure(figsize=(12, 6))
        # # plt.plot(historical_prices_actual.index, historical_prices_actual, label='Historical Prices', color='blue')
        # # plt.plot(forecast_prices_actual.index, forecast_prices_actual, label='Forecast Prices', color='red', linestyle='--')

        # # plt.plot(dataset_actual.index, dataset_actual['Close'])
        # # plt.title(f'{stock} Price Forecast')
        # # plt.xlabel('Date')
        # # plt.ylabel('Price')
        # # plt.legend()
        # # plt.grid(True)
        # # plt.show()

        # plt.figure(figsize=(12, 6))
        # plt.plot(historical_prices.index, historical_prices, label='Historical Prices', color='blue')
        # plt.plot(dataset.iloc[:-NUM_FUTURE_DAYS,].index, predicted_stock_data, label='Training Data Prediction')

        # # plt.plot(forecast_prices[viz_window(forecast_prices)].index, forecast_prices[viz_window(forecast_prices)], label='Forecast Prices (Test data)', linestyle='--')
        # # plt.plot(forecast_prices_actual[viz_window(forecast_prices_actual)].index, forecast_prices_actual[viz_window(forecast_prices_actual)], label='Forecast Prices (Wild)', linestyle='--')
        # # plt.plot(dataset[viz_window(dataset_actual)].index, dataset_actual[viz_window(dataset_actual)]['Close'],label='Historical Prices')
        # plt.title(f'{stock} Price Forecast (Training Data)')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.axvline(x = dataset.index[-1], linestyle ='--')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        #   # Plot results
        # plt.figure(figsize=(12, 6))
        # viz_window =  lambda x:  x.index > pd.to_datetime('2025-01-01')
        # # plt.plot(historical_prices.index, historical_prices, label='Historical Prices', color='blue')
        # plt.plot(forecast_prices[viz_window(forecast_prices)].index, forecast_prices[viz_window(forecast_prices)], label='Forecast Prices (Test data)', linestyle='--')
        # plt.plot(forecast_prices_actual[viz_window(forecast_prices_actual)].index, forecast_prices_actual[viz_window(forecast_prices_actual)], label='Forecast Prices (Wild)', linestyle='--')
        # plt.plot(dataset_actual[viz_window(dataset_actual)].index, dataset_actual[viz_window(dataset_actual)]['Close'],label='Historical Prices')
        # plt.title(f'{stock} Price Forecast')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.axvline(x = dataset_actual.index[-1], linestyle ='--')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


# strong_buy_fin_companies = yf.Sector("financial-services").top_companies[
#     yf.Sector("financial-services").top_companies["rating"] == "Buy"
# ]
# big_fin = strong_buy_fin_companies["market weight"] > 0.01
# big_fin_companies = strong_buy_fin_companies[big_fin]
# big_fin_companies_list = list(big_fin_companies.index)
# strong_buy_companies = yf.Sector("technology").top_companies[
#     yf.Sector("technology").top_companies["rating"] == "Strong Buy"
# ]
# strong_buy_list = list(strong_buy_companies.index)[0:2]



# full_process(big_fin_companies_list[:5], "2024-01-01", NUM_ITERS=250)

