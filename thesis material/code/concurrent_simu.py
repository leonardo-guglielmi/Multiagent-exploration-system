# ...
with Manager() as manager:
    shared_dict = manager.dict()
    processes = [Process( target=concurrent_find_goal_point
                            , args=(cf, ag, t, shared_dict, ))
                        for ag in agents ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        p.close()
    for ag in agents:
        ag.goal_point = shared_dict[ag.id]
# ...
