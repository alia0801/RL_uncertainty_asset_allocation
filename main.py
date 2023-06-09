import logging
from finrl.apps.config import *
# from tickers import *
from finrl.preprocessing.data import data_split
from gymnasium.envs.registration import register
from stock_env import StockPortfolioEnv
from stable_baselines3 import PPO
from classic_drl import DRLAgent
from config import *
from util import *
from evaluation import *
import os

if __name__ == '__main__':

    log_file_timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_filename = log_file_timestr+'.log' #datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
    logging.basicConfig(level=logging.INFO, filename='./log/'+log_filename, filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    # asset_list = DOW_30_TICKER[:4]
    
    asset_list = portfolio_list[str(portfolio_code)]
    print(asset_list)
    asset_list.sort()
    print(asset_list)

    logging.info('len_asset_list: %d'%(len(asset_list)))
    logging.info('asset_list: '+(' '.join(asset_list) ) )
    logging.info('TRAIN_START_DATE: '+TRAIN_START_DATE)
    logging.info('TRAIN_END_DATE: '+TRAIN_END_DATE)
    logging.info('TEST_START_DATE: '+TEST_START_DATE)
    logging.info('TEST_END_DATE: '+TEST_END_DATE)
    logging.info('len_TECHNICAL_INDICATORS_LIST: %d'% (len(TECHNICAL_INDICATORS_LIST) ))

    if not os.path.exists('./results/'+str(portfolio_code)+'/'+ag_name+'/'):
        os.makedirs('./results/'+str(portfolio_code)+'/'+ag_name+'/')

    processed = get_df(asset_list)
    # print(processed)
    # print(processed.columns)
    final_asset_list = list(processed.tic.unique())
    final_asset_list.sort()
    print(final_asset_list)
    logging.info('len_final_asset_list: %d'%(len(final_asset_list)))
    logging.info('final_asset_list: '+(' '.join(final_asset_list) ) )

    stock_dimension = len(processed.tic.unique())
    # state_space = 1 + 2*stock_dimension + len(TECHNICAL_INDICATORS_LIST)*stock_dimension
    # state_space = stock_dimension + len(TECHNICAL_INDICATORS_LIST)*stock_dimension
    state_space = stock_dimension + len(TECHNICAL_INDICATORS_LIST)
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    logging.info(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    train_df = data_split(processed, TRAIN_START_DATE,TRAIN_END_DATE)
    test_df = data_split(processed, TEST_START_DATE,TEST_END_DATE)
    # print(train_df)
    # print(train_df.columns)

    train_time = pd.DataFrame(train_df.time.unique(),columns=['date'])
    test_time = pd.DataFrame(test_df.time.unique(),columns=['date'])

    env_kwargs = {
        "df":train_df,
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        # "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        # "print_verbosity":5
    }
    eval_env_kwargs = {
        "df":test_df,
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        # "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        # "print_verbosity":5
    }

    if ag_name=='ppo' or 'td3':
        if ag_name=='ppo':
            ag_parm = PPO_PARAMS
        elif ag_name=='td3':
            ag_parm=TD3_PARAMS
        e_train_gym = StockPortfolioEnv(**env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        agent = DRLAgent(env = env_train)
        model = agent.get_model(ag_name,seed=100,model_kwargs = ag_parm)
        model = agent.train_model(model=model, 
                                 tb_log_name=ag_name,
                                 total_timesteps=100000)
        e_test_gym = StockPortfolioEnv(**eval_env_kwargs)
        reward_df, action_df = DRLAgent.DRL_prediction(model,e_test_gym)
        action_df.columns=final_asset_list
        print(reward_df)
        print(action_df)

        reward_df.to_csv('./results/'+str(portfolio_code)+'/'+ag_name+'/daily_reward_record.csv')
        action_df.to_csv('./results/'+str(portfolio_code)+'/'+ag_name+'/weight_record.csv')

        ann_reward,ann_stdev,mdd,stdev_week,sharpe = get_df_ABC(reward_df)
        print('test performance')
        print(ann_reward,ann_stdev,mdd,stdev_week,sharpe)
        logging.info('test performance' )
        logging.info('annual reward: %.4f'% ann_reward )
        logging.info('annual stdev: %.4f'% ann_stdev )
        logging.info('sharpe: %.4f'% sharpe )
        logging.info('mdd: %.4f'% mdd )
        logging.info('annual stdev_week: %.4f'% stdev_week )

        BH_ann_reward,BH_ann_stdev,BH_mdd,BH_week_ann_stdev,BH_sharpe = get_comb_ABC(final_asset_list,TEST_START_DATE,TEST_END_DATE)
        print('B&H performance')
        print(BH_ann_reward,BH_ann_stdev,BH_mdd,BH_week_ann_stdev,BH_sharpe)
        logging.info('B&H performance' )
        logging.info('annual reward: %.4f'% BH_ann_reward )
        logging.info('annual stdev: %.4f'% BH_ann_stdev )
        logging.info('sharpe: %.4f'% BH_sharpe )
        logging.info('mdd: %.4f'% BH_mdd )
        logging.info('annual stdev_week: %.4f'% BH_week_ann_stdev )
