from collections import namedtuple

import xpress as xp


def get_or_solution(env, optimization_maxtime):
    Vec = namedtuple('Veh', ['name', 'x', 'y'])
    Res = namedtuple('Res', ['name', 'x', 'y'])
    Order = namedtuple('Order', ['name', 'x', 'y', 'res'])

    # Driver/vehicle location
    vloc = [Vec(name='v-1', x=env.dr_x, y=env.dr_y)]

    # Restaurant list
    rlist = []
    for i in range(env.n_restaurants):
        rlist.append(Res(name='r-' + str(i + 1), x=env.res_x[i], y=env.res_y[i]))

    # Pick up list, which are restaurants associated with orders that have not been picked up yet
    # order status: 0 - inactive(not created, cancelled, delivered), 1 - open, 2 - accepted, 3 - picked-up
    plist = []  # List of pickup locations
    dlist = []  # List of delivery locations (orders, except ones that have been picked up)
    tlist = []  # List of in-transit (picked up)  orders
    A = []  # Index list of accepted orders
    r = []  # reward list
    lt_oa = []  # late time window for open/accepted orders
    lt_it = []  # late time window for in-transit orders

    noa = 0  # Number of open or accepted orders
    nit = 0  # Numbrt of in-transit (picked up) orders

    opt_env_action_map_open = {}
    opt_env_action_map_intransit = {}

    for i in range(len(env.o_status)):
        if env.o_status[i] == 1:  # Open order
            noa += 1
            assigned_res = env.o_res_map[i]
            plist.append(rlist[assigned_res])
            dlist.append(Order(name='o-' + str(noa), x=env.o_x[i], y=env.o_y[i], res=assigned_res + 1))
            r.append(env.reward_per_order[i])
            opt_env_action_map_open[noa] = i + 1
            lt_oa.append(env.order_promise - env.o_time[i])

        elif env.o_status[i] == 2:  # Accepted order
            noa += 1
            opt_env_action_map_open[noa] = i + 1
            assigned_res = env.o_res_map[i]
            plist.append(rlist[assigned_res])
            dlist.append(Order(name='o-' + str(noa), x=env.o_x[i], y=env.o_y[i], res=assigned_res + 1))
            A.append(noa)
            r.append(env.reward_per_order[i])
            lt_oa.append(env.order_promise - env.o_time[i])
        elif env.o_status[i] == 3:  # In-transit (picked up) order
            nit += 1
            opt_env_action_map_intransit[nit] = i + 1
            tlist.append(Order(name='t-' + str(nit), x=env.o_x[i], y=env.o_y[i], res=0))
            lt_it.append(env.order_promise - env.o_time[i])

    # Creation of index sets/lists
    nodes = vloc + plist + dlist + tlist + rlist
    n = len(plist)
    P = list(range(1, n + 1))  # Pick-up indices
    D = list(range(n + 1, 2 * n + 1))  # Delivery indices
    T = list(range(2 * len(P) + 1, 2 * len(P) + 1 + len(tlist)))  # In-transit indices
    R = list(range(2 * len(P) + len(T) + 1, 2 * len(P) + len(T) + 1 + len(rlist)))  # Restaurant indices to return
    N = list(range(len(nodes)))  # All indices

    # Create optimization parameters
    E = [(i, j) for i in N for j in N]  # Edges
    m = env.penalty_per_timestep + env.penalty_per_move  # Cost per mile, can be parametrized
    U = env.driver_capacity  # Driver capacity
    M = 99999  # A very big number
    q = [len(T)] + [1] * len(plist) + [-1] * len(dlist) + [-1] * len(tlist) + [0] * len(rlist)  # Capacity usage
    C = {(i, j): abs(nodes[i].x - nodes[j].x) + abs(nodes[i].y - nodes[j].y) for (i, j) in E}  # Distance matrix
    t = 1  # Time to travel 1 mile in minutes
    d = 1  # Service time in minutes

    # MIP Model
    # whether the vehicle uses the arc from i to j
    x = {(i, j): xp.var(name='x_{0}_{1}'.format(i, j), vartype=xp.binary) for (i, j) in E}
    # track the capacity ussage as of node j
    Q = {j: xp.var(name='Q_{0}'.format(j)) for j in N}
    # track the time as of node j
    B = {j: xp.var(name='B_{0}'.format(j)) for j in N}
    # whether the order is accepted or not
    y = {i: xp.var(name='y_{0}'.format(i), vartype=xp.binary) for i in P}

    # constraints in VRP baseline formulation
    leave = [xp.Sum(x[i, j] for j in N) == y[i] for i in P]
    pickup = [xp.Sum(x[i, j] for j in N) - xp.Sum(x[i + n, j] for j in N) == 0 for i in P]
    accepted = [y[i] == 1 for i in A]
    in_transit = [xp.Sum(x[i, j] for j in N) == 1 for i in T]  # Add in-transit here
    start = [xp.Sum(x[0, j] for j in N) == 1]
    # end = [xp.Sum(x[j, i] for j in N for i in R) == 1]
    end = [xp.Sum(x[j, i] for j in N[:-len(R)] for i in R) == 1]
    # stop = [xp.Sum(x[i, j] for i in R for j in N ) == 0] # Remove this
    flow = [xp.Sum(x[j, i] for j in N[:-len(R)]) - xp.Sum(x[i, j] for j in N) == 0 for i in P + D + T]
    # capacity constraints
    cap_track = [Q[j] >= Q[i] + q[j] - M * (1 - x[i, j]) for i in N for j in N]
    cap_lb = [max(0, q[i]) <= Q[i] for i in N]
    cap_ub = [Q[i] <= min(U, U + q[i]) for i in N]
    # time constraints
    time_track = [B[j] >= B[i] + d * (i != 0) + C[i, j] * t - M * (1 - x[i, j]) for i in N for j in N]
    precedence = [B[i] + C[i, i + n] * t - M * (1 - y[i]) <= B[i + n] for i in P]
    timetoaccept = [B[0] == d * xp.Sum(y[i] for i in P if i not in A)]
    timewindow_oa = [B[i] <= lt_oa[i - 1 - len(P)] for i in D]
    timewindow_it = [B[i] <= lt_it[i - 1 - 2 * len(P)] for i in T]
    # timewindow_oa = [B[i] <= lt_oa[i - 1 - len(P)] - xp.Sum(y[i] for i in P) + len(A) for i in D]
    # timewindow_it = [B[i] <= lt_it[i - 1 - 2*len(P)] - xp.Sum(y[i] for i in P) + len(A) for i in T]

    p = xp.problem()

    p.addVariable(x, Q, B, y)
    p.addConstraint(leave, pickup, accepted, in_transit, start, end, flow, cap_track, cap_lb, cap_ub, time_track,
                    precedence, timewindow_oa, timewindow_it, timetoaccept)

    # objective function.
    # @annaluo: what are the last two terms for?
    p.setObjective(xp.Sum(r[i - 1] * y[i] for i in P)
                   - m * xp.Sum(C[i, j] * x[i, j] for (i, j) in E)
                   - xp.Sum(Q[i] for i in N) * 0.0001
                   - xp.Sum(B[i] for i in N) * 0.0001
                   , sense=xp.maximize)

    p.controls.maxtime = -optimization_maxtime  # negative for "stop even if no solution is found"

    # print(C)
    print("P", P)
    print("D", D)
    print("T", T)
    print("R", R)
    print("N", N)

    print()

    p.solve()

    print("problem status: ", p.getProbStatusString())
    if p.getProbStatusString() == 'mip_infeas':
        p.iisall()
        print("there are ", p.attributes.numiis, " iis’s")
        miisrow = []
        miiscol = []
        constrainttype = []
        colbndtype = []
        duals = []
        rdcs = []
        isolationrows = []
        isolationcols = []
        # get data for the first IIS
        p.getiisdata(1, miisrow, miiscol, constrainttype, colbndtype,
                     duals, rdcs, isolationrows, isolationcols)
        print("iis data:", miisrow, miiscol, constrainttype, colbndtype,
              duals, rdcs, isolationrows, isolationcols)
        # Another way to check IIS isolations
        print("iis isolations:", p.iisisolations(1))
        rowsizes = []
        colsizes = []
        suminfeas = []
        numinfeas = []
        print("iisstatus:", p.iisstatus(rowsizes, colsizes, suminfeas, numinfeas))
        print("vectors:", rowsizes, colsizes, suminfeas, numinfeas)

        p.write("vrp", "l")
        with open("vrp.lp") as f:
            for l in f:
                print(l)

    sol = p.getSolution()

    def ix_to_act(ix):
        if ix == 0:
            action = 0
        elif ix in P:  # Pickup action
            action = opt_env_action_map_open[ix] + env.n_orders
        elif ix in D:  # Deliver action
            action = opt_env_action_map_open[ix - len(P)] + 2 * env.n_orders
        elif ix in T:  # Deliver an in transit item
            action = opt_env_action_map_intransit[ix - 2 * len(P)] + 2 * env.n_orders
        elif ix in R:  # Return to restaurant action
            action = ix - 2 * len(P) - len(T) + 3 * env.n_orders
        else:
            raise Exception('Unrecognized action in optimization solution: {}'.format(ix))
        return action

    y_sol = [i for i in P if sol[p.getIndex(y[i])] > 0.5]  # Accepted orders
    x_sol = {i: j for (i, j) in E if sol[p.getIndex(x[i, j])] > 0.5}  # Travel sequence

    print()
    print("Map:", opt_env_action_map_open)
    print()
    print("B values")
    print()
    # print([(i, p.getIndex (B[i]), ) for i in N])
    print([(ix_to_act(i), sol[p.getIndex(B[i])], lt_oa[i - 1 - len(P)] if i in D else None) for i in N])

    print("Y solution:", y_sol)
    print()
    print("X solution:", x_sol)
    print()
    # print("C vectors:", [(ix_to_act(i), ix_to_act(j), C[i,j]) for i in N for j in N] )
    # print()

    ### Convert solutions into action sequences
    # First accept all orders that we have decided to accept.
    # Do not include already accepted orders
    actions = [opt_env_action_map_open[y] for y in y_sol if env.o_status[opt_env_action_map_open[y] - 1] == 1]
    times = [(i, env.clock + i + 1) for i in
             range(len([y for y in y_sol if env.o_status[opt_env_action_map_open[y] - 1] == 1]))]

    current = 0  # Start with the driver location to extract the route
    while current in x_sol:
        next_stop = x_sol[current]
        action = ix_to_act(next_stop)
        actions.append(action)  # Shift the actions to align with env action space
        b_time = env.clock + sol[p.getIndex(B[next_stop])]
        times.append((action, b_time))
        current = next_stop

    print("Times:", times)
    return actions


def whether_get_opt_sln(env, prev_o_status, prev_o_xy, current_action_plan):
    ''' Ask for a new solution if
        1 - A new order has arrived
        2 - An order has expired (not delivered)
        @annaluo 3 - All actions executed
    '''

    # get_new_solution = False
    # If any new orders has arrived
    new_orders = [o for o in range(env.n_orders) if prev_o_status[o] == 0 and env.o_status[o] == 1]
    if new_orders:
        print("New order has arrived!")
        return True

    if len(current_action_plan) == 0:
        return True

    for o in range(env.n_orders):
        # If we plan to accept but the order has changed
        # Add here if the restaurant has changed as well
        if o + 1 in current_action_plan and (((env.o_x[o], env.o_y[o]) != prev_o_xy[o])):
            print("Order", o + 1, "we were planning to accept has changed. Resolving the opt problem.")
            return True

    return False


def execute_plan(env, action_plan):
    no = env.n_orders
    new_action_plan = []
    for i in range(len(action_plan)):
        next_action = action_plan[i]
        # print(next_action)
        if next_action == 0:
            action = next_action
            if i < len(action_plan) - 1:
                new_action_plan = action_plan[i + 1:]
            break

        elif next_action <= no:  # Accept order
            if env.o_status[next_action - 1] == 1:  # Order is open
                action = next_action
                # if i < len(action_plan) - 1:
                #    new_action_plan = action_plan[i + 1:]
                if i < len(action_plan):
                    new_action_plan = action_plan[i:]
                break

        elif next_action <= 2 * no:  # Pickup an order
            if env.o_status[next_action - 1 - no] == 2:  # Order is ready to be picked up
                action = next_action
                if i < len(action_plan):
                    new_action_plan = action_plan[i:]
                break

        elif next_action <= 3 * no:  # Deliver an order
            if env.o_status[next_action - 1 - 2 * no] == 3:  # Order is ready to be delivered
                action = next_action
                if i < len(action_plan):
                    new_action_plan = action_plan[i:]
                break
        else:  # Go back to a restaurant
            # @TODO: How to determine that this has been achieved? Check the locations.
            action = next_action
            if (env.dr_x, env.dr_y) != (env.res_x[action - 3 * no - 1], env.res_y[action - 3 * no - 1]):
                new_action_plan = [action]
            break

    return action, new_action_plan
