def get_dataset_order(dataset_id):
    task_list = [
        "yelp",
        "amazon",
        "dbpedia",
        "yahoo",
        "agnews",
        "MNLI",
        "QQP",
        "RTE",
        "SST-2",
        "WiC",
        "CB",
        "COPA",
        "BoolQA",
        "MultiRC",
        "IMDB"
    ]
    if dataset_id == 1:
        dataset_order = [
            "mnli",
            "cb",
            "wic",
            "copa",
            "qqp",
            "boolqa",
            "rte",
            "imdb",
            "yelp",
            "amazon",
            "sst-2",
            "dbpedia",
            "agnews",
            "multirc",
            "yahoo"
            ]
        
    elif dataset_id == 2:
        dataset_order = [
            "multirc",
            "boolqa",
            "wic",
            "mnli",
            "cb",
            "copa",
            "qqp",
            "rte",
            "imdb",
            "sst-2",
            "dbpedia",
            "agnews",
            "yelp",
            "amazon",
            "yahoo"
        ]
    elif dataset_id == 3:
        dataset_order = [
            "yelp",
            "amazon",
            "mnli",
            "cb",
            "copa",
            "qqp",
            "rte",
            "imdb",
            "sst-2",
            "dbpedia",
            "agnews",
            "yahoo",
            "multirc",
            "boolqa",
            "wic"
        ]
    elif dataset_id == 4:
        dataset_order = [
            "dbpedia",
            "amazon",
            "yahoo",
            "agnews",
        ]
    elif dataset_id == 5:
        dataset_order = [
            "dbpedia",
            "amazon",
            "agnews",
            "yahoo",
        ]
    elif dataset_id == 6:
        dataset_order = [
            "yahoo",
            "amazon",
            "agnews",
            "dbpedia",
        ]
    elif dataset_id == 7:
        dataset_order = [
            "task1572",
            "task363",
            "task1290",
            "task181",
            "task002",
            "task1510",
            "task639",
            "task1729",
            "task073",
            "task1590",
            "task748",
            "task511",
            "task591",
            "task1687",
            "task875"
            ]
        
    elif dataset_id == 8:
        dataset_order = [
            "task748",
            "task073",
            "task1590",
            "task639",
            "task1572",
            "task1687",
            "task591",
            "task363",
            "task1510",
            "task1729",
            "task181",
            "task511",
            "task002",
            "task1290",
            "task875"
        ]
    else:
        raise

    return dataset_order