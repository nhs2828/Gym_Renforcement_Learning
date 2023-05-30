cfgDDGP = {
    "len_buffer" : 1000000,
    "batch" : 64,
    "gamma" : 0.99,
    "tau" : 0.001,
    "explore": 1.0,
    "explore_min": 0.01,
    "explore_decay": 0.995,
    "lr" : {
        "critic" : 0.000025,
        "actor" : 0.00025
    },
    "hidden_layer" : {
        "critic" : [400,300],
        "actor" : [400, 300]
    }
}