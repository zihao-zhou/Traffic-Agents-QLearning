from traci_backend import TraciBackend
import time

if __name__ == "__main__":
    backend = TraciBackend()
    backend.start()
    lanes = backend.get_lane_ids()
    lanes_cnt = len(lanes)
    print("Lane ids are: {}".format(lanes))
    for i in range(1, 100):
        backend.simulate_step()
        if i == 1:
#             print(backend.get_light_definition("0d"))
            print(backend.get_light_ids())

#         if i % 20 == 0:
#             backend.set_light_phase("0d", 1)

        vehicle_ids = backend.get_vehicle_ids()
#         cur_state = backend.get_cur_light_state("0d")

        avg_passed_vehicle = 0.0
        print("Time {}: Vehicle # are: {}\nTraffic light".format(i, len(vehicle_ids)))
        for lane in lanes:
            avg_passed_vehicle += (backend.get_vehicle_cnt(lane) - backend.get_halt_vehicle_cnt(lane)) / float(lanes_cnt)

        print("Average passed vehicle nuber on {} lanes are: {}".format(lanes_cnt, avg_passed_vehicle))

        time.sleep(0.2)

    backend.close()
    exit()
