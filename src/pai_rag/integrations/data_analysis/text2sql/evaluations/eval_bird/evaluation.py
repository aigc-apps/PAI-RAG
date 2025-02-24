import sys
import os
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut


def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents


# def result_callback(result):
#     exec_result.append(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    # print("predicted_sql:", predicted_sql)
    # print("ground_truth:", ground_truth)
    # print("db_path:", db_path)
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(
            meta_time_out, execute_sql, args=(predicted_sql, ground_truth, db_place)
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [("timeout",)]
        res = 0
    except Exception:
        result = [("error",)]  # possibly len(query) > 512 or not executable
        res = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {"sql_idx": idx, "res": res}
    # print("execute_model_result:", result)
    return result


def package_sqls(sql_path, db_root_path, mode="gpt", data_mode="dev"):
    clean_sqls = []
    db_path_list = []

    if mode == "gpt":
        # sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        with open(sql_path, "r", encoding="utf-8") as file:
            sql_data = json.load(file)
        for idx, sql_str in sql_data.items():
            # if type(sql_str) == str:
            if isinstance(sql_str, str):
                sql, db_name = sql_str.split("\t----- bird -----\t")
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path = os.path.join(db_root_path, db_name, f"{db_name}.sqlite")
            db_path_list.append(db_path)
            # db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == "gt":
        sqls = open(sql_path + data_mode + ".sql")
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split("\t")
            clean_sqls.append(sql)
            db_path = os.path.join(db_root_path, db_name, f"{db_name}.sqlite")
            db_path_list.append(db_path)
            # db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list


# def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
#     pool = mp.Pool(processes=num_cpus)
#     for i,sql_pair in enumerate(sqls):

#         predicted_sql, ground_truth = sql_pair
#         pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
#     pool.close()
#     pool.join()


def run_sqls_parallel(sqls, db_places, exec_result, num_cpus=1, meta_time_out=30.0):
    with mp.Manager() as manager:
        shared_exec_result = manager.list()  # 创建一个可以在多个进程中共享的列表

        def result_callback(result):
            shared_exec_result.append(result)

        with mp.Pool(processes=num_cpus) as pool:
            for i, sql_pair in enumerate(sqls):
                predicted_sql, ground_truth = sql_pair
                pool.apply_async(
                    execute_model,
                    args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
                    callback=result_callback,
                )
            pool.close()
            pool.join()

        # 将共享列表转换为普通列表并返回
        exec_result.extend(list(shared_exec_result))


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents[0:num_queries]):
        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])

        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])

        if content["difficulty"] == "challenging":
            challenging_results.append(exec_results[i])

    if len(simple_results) != 0:
        simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results)
    else:
        simple_acc = 0
    if len(moderate_results) != 0:
        moderate_acc = sum([res["res"] for res in moderate_results]) / len(
            moderate_results
        )
    else:
        moderate_acc = 0
    if len(challenging_results) != 0:
        challenging_acc = sum([res["res"] for res in challenging_results]) / len(
            challenging_results
        )
    else:
        challenging_acc = 0
    all_acc = sum(results) / num_queries
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


def print_data(score_lists, count_lists):
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))

    print(
        "======================================    ACCURACY    ====================================="
    )
    print(
        "{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format("accuracy", *score_lists)
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--data_mode", type=str, required=True, default="dev")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--mode_gt", type=str, default="gt")
    args_parser.add_argument("--mode_predict", type=str, default="gpt")
    args_parser.add_argument("--difficulty", type=str, default="simple")
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode=args.mode_predict,
        data_mode=args.data_mode,
    )
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path, args.db_root_path, mode="gt", data_mode=args.data_mode
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
    )
    exec_result = sort_results(exec_result)

    print("start calculate")
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result, args.diff_json_path
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print(
        "==========================================================================================="
    )
    print("Finished evaluation")
