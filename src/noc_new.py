import simpy
import logging
import contextlib
from enum import IntEnum
from src.arch_config import LinkConfig, RouterConfig, NoCConfig
from src.sim_type import Data, Message, ceil, Slice, Direction, DataType, MsgType
from src.common import MonitoredResource

logger = logging.getLogger("NoC")

# class Link:
#     def __init__(self, env, config: LinkConfig):
#         self.env = env
#         self.width = config.width
#         self.delay = config.delay
#         # self.store = simpy.Store(env)

#     def transmit(self, size):
#         transmission_time = ceil(size, self.width)
#         latency = self.delay + transmission_time

#         # self.store.put(size)
#         yield self.env.timeout(latency)
#         # self.store.get()

#     def change_width(self, times):
#         self.width *= times

class Link:
    def __init__(self, env: simpy.Environment, config: LinkConfig):
        self.env = env
        #self.config = config
        #self.delay = config.delay
        self.width = config.width
        self.delay = config.delay
        self.store = simpy.PriorityStore(env)
        self.delay_factor = 1
        self.hop = 0
        self.linkentry = MonitoredResource(env,capacity=1)
        self.tag = False

    def bind(self, idx1, idx2, tag):
        self.corefrom = idx1
        self.coreto = idx2
        self.tag = tag
    def calc_latency(self, msg):
        #calc latency:
        slice = Slice(tensor_slice=msg.data.tensor_slice)
        if msg.ins.data_type == DataType.FEAT:
            transmission_time = ceil(slice.size()*msg.ins.feat_precision, self.width)
        else:
            transmission_time = ceil(slice.size()*msg.ins.para_precision, self.width)
        latency = self.delay + transmission_time
        latency = latency * self.delay_factor

        self.hop += slice.size()/64
        # 对于数据包,记录了路由路径中每个link的ready_run_time
        msg.ins.record.ready_run_time.append(self.env.now)
        yield self.linkentry.execute("SEND"+str(msg.data.index),latency,msg.ins,attributes=msg.dst)
        self.store.put(msg)
          
    # def calc_latency(self, msg: Message):
    #     latency = 0
    #     if msg.msg_type == MsgType.MEM_REQUEST:
    #         # 请求消息：固定延迟
    #         latency = self.delay
    #     elif msg.msg_type == MsgType.DATA:
    #         # 数据消息：延迟与数据大小有关
    #         if msg.data is None:
    #             raise ValueError("Message with type DATA has no data payload.")
    #         slice = Slice(tensor_slice=msg.data.tensor_slice)
    #         size = slice.size()
            
    #         # latency = size / (self.config.width / 8) + self.delay
    #         latency = ceil(size, self.config.width) + self.delay
    #     else:
    #         # 对于其他未知的消息类型，可以返回一个默认延迟或抛出错误
    #         latency = self.delay

    #     # 模拟传输延迟
    #     yield self.env.timeout(latency)
        
    #     # 延迟结束后，将消息放入store，使其在链路另一端可用
    #     put_event = self.store.put(msg)
    #     yield put_event
        
    #     # 调试信息：检查消息是否成功进入store
    #     if hasattr(self, 'store') and hasattr(self.store, 'items'):
    #         print(f"Time {self.env.now:.2f}: Link calc_latency finished, store now has {len(self.store.items)} items")
    #         logger.debug(f"Time {self.env.now:.2f}: Link calc_latency finished, store now has {len(self.store.items)} items")
            
    #         # 如果是发送到Memory Cores的消息，输出详细信息
    #         if hasattr(msg, 'dst') and msg.dst in [0, 1, 2, 3, 12, 13, 14, 15]:
    #             print(f"Time {self.env.now:.2f}: Message for Core {msg.dst} put into store, store length: {len(self.store.items)}")
    #             logger.debug(f"Time {self.env.now:.2f}: Message for Core {msg.dst} put into store, store length: {len(self.store.items)}")
    #             print(f"Time {self.env.now:.2f}: put_event.triggered = {put_event.triggered}")
    #             logger.debug(f"Time {self.env.now:.2f}: put_event.triggered = {put_event.triggered}")
            
    def run(self):
        while True:
            msg = yield self.store.get()

    def put(self, msg):
        return self.env.process(self.calc_latency(msg))
    
    # 不带延迟的插入，处理block外的数据包时使用
    def insert(self, msg):
        yield self.store.put(msg)

    def get(self):
        return self.store.get()
    
    def len(self):
        return len(self.store.items)

    def change_delay(self, times):
        self.delay_factor *= times

    def recover_delay(self, times):
        self.delay_factor /= times

class Router:
    def __init__(self, env, config: RouterConfig, id: int, x: int, y:int, noc_ref=None):
        self.env = env
        self.x = x
        self.y = y
        self.id = id
        self.noc = noc_ref  # 新增：NoC全局引用

        self.type = config.type
        self.vc = config.vc

        self.core_in, self.core_out = None, None
        self.north_in, self.north_out = None, None
        self.south_in, self.south_out = None, None
        self.east_in, self.east_out = None, None
        self.west_in, self.west_out = None, None

        self.env.process(self.run())

    def bound_with_north(self, north_in, north_out):
        self.north_in = north_in
        self.north_out = north_out

    def bound_with_south(self, south_in, south_out):
        self.south_in = south_in
        self.south_out = south_out

    def bound_with_east(self, east_in, east_out):
        self.east_in = east_in
        self.east_out = east_out

    def bound_with_west(self, west_in, west_out):
        self.west_in = west_in
        self.west_out = west_out

    def bound_with_core(self, core_in, core_out):
        self.core_in = core_in
        self.core_out = core_out

    def router_fail(self, times):
        self.north_in.change_delay(times)
        self.north_out.change_delay(times)

        self.south_in.change_delay(times)
        self.south_out.change_delay(times)

        self.east_in.change_delay(times)
        self.east_out.change_delay(times)

        self.west_in.change_delay(times)
        self.west_out.change_delay(times)

    def router_recover(self, times):
        self.north_in.recover_delay(times)
        self.north_out.recover_delay(times)
        
        self.south_in.recover_delay(times)
        self.south_out.recover_delay(times)

        self.east_in.recover_delay(times)
        self.east_out.recover_delay(times)
        
        self.west_in.recover_delay(times)
        self.west_out.recover_delay(times)

    def route_with_strategy(self, msg: Message, strategy="hop_by_hop"):
        """
        根据指定策略进行路由
        
        Args:
            msg: 要路由的消息
            strategy: 路由策略 ("hop_by_hop" 或 "wormhole")
        """
        if strategy == "wormhole":
            # 使用Wormhole路由
            yield self.env.process(self.wormhole_route_via_noc(msg))
        else:
            # 使用传统的hop-by-hop路由
            if msg.dst != self.id:
                next_dir, next_router = self.calculate_next_router(msg.dst)
                logger.info(f"Time {self.env.now:.2f}: Router{self.id} start hop-by-hop routing data{msg.data.index} to router{next_router}(dst:{msg.dst}).")
                yield self.env.process(self.one_hop_route(msg, next_dir, next_router))

    def one_hop_route(self, msg: Message, next_dir, next_router):
        match next_dir:
            case Direction.NORTH:
                yield self.north_out.put(msg)
            case Direction.SOUTH:
                yield self.south_out.put(msg)
            case Direction.EAST:
                yield self.east_out.put(msg)
            case Direction.WEST:
                yield self.west_out.put(msg)

        # 记录日志
        if msg.data:
            logger.info(f"Time {self.env.now:.2f}: Router {self.id} finish sending data {msg.data.index} to Router {next_router} (dst: {msg.dst}).")
        else:
            logger.info(f"Time {self.env.now:.2f}: Router {self.id} finish sending {msg.msg_type.name} to Router {next_router} (dst: {msg.dst}).")

    def route_core(self, msg):
        logger.info(f"Router{self.id} routing to core. Store object: {self.core_out}")
        yield self.core_out.put(msg)
        if msg.data:
            logger.info(f"Time {self.env.now:.2f}: Finish putting data {msg.data.index} to Core {self.id}")
        else:
            logger.info(f"Time {self.env.now:.2f}: Finish putting {msg.msg_type.name} to Core {self.id}")
 
    def process_hop_by_hop(self, message):
        """Helper function to process a single message via hop-by-hop routing."""
        if message.should_deliver_to_core(self.id):
            if message.data:
                logger.info(f"Time {self.env.now:.2f}: Hop-by-hop message {message.data.index} delivered to Router {self.id} for core.")
            else:
                logger.info(f"Time {self.env.now:.2f}: Hop-by-hop {message.msg_type.name} delivered to Router {self.id} for core.")
            self.env.process(self.route_core(message))
        
        if message.dst != self.id:
            next_dir, next_router = self.calculate_next_router(message.dst)
            if message.data:
                logger.info(f"Time {self.env.now:.2f}: Router {self.id} start hop-by-hop forwarding data {message.data.index} to Router {next_router} (dst: {message.dst}).")
            else:
                logger.info(f"Time {self.env.now:.2f}: Router {self.id} start hop-by-hop forwarding {message.msg_type.name} to Router {next_router} (dst: {message.dst}).")
            self.env.process(self.one_hop_route(message, next_dir, next_router))

    def run(self):
        all_possible_channels = [
            (self.north_in, "north"), (self.south_in, "south"),
            (self.east_in, "east"), (self.west_in, "west"),
            (self.core_in, "core")
        ]
        all_channels = [channel for channel in all_possible_channels if channel[0] is not None]

        while True:
            with contextlib.ExitStack() as stack:
                events = [stack.enter_context(channel[0].get()) for channel in all_channels]
                yield self.env.any_of(events)

                for idx, event in enumerate(events):
                    if event.triggered:
                        msg = event.value
                        channel_obj, channel_name = all_channels[idx]
                        
                        # Apply wormhole logic ONLY for the first message from the core if flagged
                        is_from_core = (channel_name == "core")
                        if is_from_core:
                            logger.debug(f"Time {self.env.now:.2f}: Router {self.id} received a message from its core_in link.")
                        
                        if is_from_core and hasattr(msg, 'route_strategy') and msg.route_strategy == 'wormhole':
                            log_payload = f"data {msg.data.index}" if msg.data else f"{msg.msg_type.name}"
                            logger.info(f"Time {self.env.now:.2f}: Router {self.id} begins wormhole request for {log_payload} from core.")
                            self.env.process(self.wormhole_route_via_noc(msg))
                        else:
                            self.process_hop_by_hop(msg)

                        # Restore the original logic: drain the rest of the channel using hop-by-hop
                        while channel_obj.len() > 0:
                            next_msg = yield channel_obj.get()
                            self.process_hop_by_hop(next_msg)

    #模拟其它流量造成的网络拥堵
    def trans(self,start_time,link,flow):
        yield self.env.timeout(start_time)
        yield self.env.process(link.transmit(flow))
        
        
    def addtion_flow(self,env,linklist,timelist,flowlist):
        #加入额外的流量，用list来记录，linklist是List[(link)],timelist是List[(time)]
        lenth=len(linklist)
        assert lenth==len(timelist)
        for i in range(lenth):
            start_time=timelist[i]
            link=linklist[i]
            flow=flowlist[i]    
            env.process(self.trans(start_time,link,flow))
        
        

    def calculate_next_router(self, target_id):
        if self.type == "XY":
            # switch id
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            # X first
            if now_x != tar_x:
                if tar_x > now_x:
                    return Direction.EAST, self.to_x(now_x + 1, now_y)
                else:
                    return Direction.WEST, self.to_x(now_x - 1, now_y)
        
            # then Y
            if now_y != tar_y:
                if tar_y > now_y:
                    return Direction.NORTH, self.to_x(now_x, now_y + 1)
                else:
                    return Direction.SOUTH, self.to_x(now_x, now_y - 1)
        else:
            return target_id


    def full_path_route(self, target_id, strategy="XY"):
        """
        规划从当前路由器到目标路由器的完整路径（支持多种策略）
        
        Args:
            target_id: 目标路由器的ID
            strategy: 路由策略 ("XY", "YX", "balanced")
            
        Returns:
            path: 路径上所有路由器ID的列表，包括源和目的地
            directions: 对应的方向列表
            total_hops: 总跳数
        """
        # 如果目标就是当前路由器，直接返回
        if target_id == self.id:
            return [self.id], [], 0
        
        # 获取当前和目标坐标
        src_x, src_y = self.to_xy(self.id)
        dst_x, dst_y = self.to_xy(target_id)
        
        # 计算曼哈顿距离
        manhattan_distance = abs(dst_x - src_x) + abs(dst_y - src_y)
        
        path = [self.id]
        directions = []
        current_x, current_y = src_x, src_y
        
        if strategy == "XY":
            # XY策略：先X后Y
            # X方向移动
            while current_x != dst_x:
                if dst_x > current_x:
                    current_x += 1
                    directions.append(Direction.EAST)
                else:
                    current_x -= 1
                    directions.append(Direction.WEST)
                path.append(self.to_x(current_x, current_y))
            
            # Y方向移动
            while current_y != dst_y:
                if dst_y > current_y:
                    current_y += 1
                    directions.append(Direction.NORTH)
                else:
                    current_y -= 1
                    directions.append(Direction.SOUTH)
                path.append(self.to_x(current_x, current_y))
                
        elif strategy == "YX":
            # YX策略：先Y后X
            # Y方向移动
            while current_y != dst_y:
                if dst_y > current_y:
                    current_y += 1
                    directions.append(Direction.NORTH)
                else:
                    current_y -= 1
                    directions.append(Direction.SOUTH)
                path.append(self.to_x(current_x, current_y))
            
            # X方向移动
            while current_x != dst_x:
                if dst_x > current_x:
                    current_x += 1
                    directions.append(Direction.EAST)
                else:
                    current_x -= 1
                    directions.append(Direction.WEST)
                path.append(self.to_x(current_x, current_y))
                
        elif strategy == "balanced":
            # 平衡策略：交替进行X和Y移动，尽量平衡负载
            x_steps = abs(dst_x - src_x)
            y_steps = abs(dst_y - src_y)
            x_dir = 1 if dst_x > src_x else -1
            y_dir = 1 if dst_y > src_y else -1
            
            x_moved, y_moved = 0, 0
            
            while x_moved < x_steps or y_moved < y_steps:
                # 决定下一步移动方向
                if x_moved < x_steps and y_moved < y_steps:
                    # 两个方向都还需要移动，选择剩余步数更多的方向
                    if (x_steps - x_moved) >= (y_steps - y_moved):
                        # X方向移动
                        current_x += x_dir
                        x_moved += 1
                        directions.append(Direction.EAST if x_dir > 0 else Direction.WEST)
                    else:
                        # Y方向移动
                        current_y += y_dir
                        y_moved += 1
                        directions.append(Direction.NORTH if y_dir > 0 else Direction.SOUTH)
                elif x_moved < x_steps:
                    # 只需要X方向移动
                    current_x += x_dir
                    x_moved += 1
                    directions.append(Direction.EAST if x_dir > 0 else Direction.WEST)
                else:
                    # 只需要Y方向移动
                    current_y += y_dir
                    y_moved += 1
                    directions.append(Direction.NORTH if y_dir > 0 else Direction.SOUTH)
                
                path.append(self.to_x(current_x, current_y))
        
        return path, directions, manhattan_distance

    # to1D id, 0-indexed
    def to_x(self, x, y):
        return x * self.y + y
    
    # to2D id, 0-indexed
    def to_xy(self, id):
        x = id // self.y
        y = id % self.y
        return x, y

    def wormhole_route_via_noc(self, msg: Message):
        """
        通过NoC全局管理器执行Wormhole路由
        """
        if self.noc is None:
            raise RuntimeError("NoC reference not available for wormhole routing")
        
        log_payload = f"data {msg.data.index}" if msg.data else f"{msg.msg_type.name}"
        logger.info(f"Time {self.env.now:.2f}: Router {self.id} initiating wormhole routing for {log_payload} to Router {msg.dst}")
        yield self.env.process(self.noc.wormhole_send(self.id, msg))

class NoC:
    def __init__(self, env, config: NoCConfig):
        self.env = env
        self.x = config.x
        self.y = config.y
        self.router_config = config.router
        self.link_config = config.link
        self.r2r_links = []
        self.routers = []

    # connections between routers
    def build_connection(self):
        for id in range(self.x * self.y):
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self))

        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                # connect with the east Router
                if row < self.x - 1:
                    east_router_id = (row + 1) * self.y + col
                    
                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    link2.bind((row,col), (row+1,col), True)
                    link1.bind((row+1,col), (row,col), True)

                    self.routers[router_id].bound_with_east(link1, link2)
                    self.routers[east_router_id].bound_with_west(link2, link1)

                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)

                # connect with the south Router
                if col > 0:
                    south_router_id = row * self.y + (col - 1)

                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    link2.bind((row,col), (row,col-1), True)
                    link1.bind((row,col-1), (row,col), True)

                    self.routers[router_id].bound_with_south(link1, link2)
                    self.routers[south_router_id].bound_with_north(link2, link1)
                    
                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)
                    
        return self

    def get_link_between_routers(self, router1_id, router2_id):
        """
        获取两个相邻Router之间的链路
        
        Args:
            router1_id: 源路由器ID
            router2_id: 目标路由器ID
            
        Returns:
            link: 连接两个路由器的链路对象
        """
        router1 = self.routers[router1_id]
        
        # 计算相对位置
        r1_x, r1_y = router1.to_xy(router1_id)
        r2_x, r2_y = router1.to_xy(router2_id)
        
        # 根据相对位置确定链路
        if r2_x == r1_x + 1 and r2_y == r1_y:  # router2在router1的东方
            return router1.east_out
        elif r2_x == r1_x - 1 and r2_y == r1_y:  # router2在router1的西方
            return router1.west_out
        elif r2_x == r1_x and r2_y == r1_y + 1:  # router2在router1的北方
            return router1.north_out
        elif r2_x == r1_x and r2_y == r1_y - 1:  # router2在router1的南方
            return router1.south_out
        else:
            raise ValueError(f"Router {router1_id} and {router2_id} are not adjacent")

    def wormhole_send(self, src_router_id, msg):
        """
        在NoC级别实现Wormhole路由
        
        Args:
            src_router_id: 源路由器ID
            msg: 要传输的消息
        """
        # 紧急修复：处理源和目的地相同的情况
        if src_router_id == msg.dst:
            target_router = self.routers[msg.dst]
            log_payload = f"data {msg.data.index}" if msg.data else f"{msg.msg_type.name}"
            logger.info(f"Time {self.env.now:.2f}: Wormhole source Router{src_router_id} is same as destination. Delivering {log_payload} locally.")
            if hasattr(target_router, 'core_in') and target_router.core_in:
                # 对于本地投递，没有网络延迟。可以立即放入
                # 使用 timeout(0) 是为了确保它在下一个仿真delta中发生，避免同步执行问题
                yield self.env.timeout(0)
                target_router.core_in.store.put(msg)
            else:
                logger.warning(f"Time {self.env.now:.2f}: Local delivery target Router{msg.dst} has no core_in, message may be lost.")
            return  # 提前退出

        from src.sim_type import Slice, DataType
        from src.sim_type import ceil
        
        src_router = self.routers[src_router_id]
        
        # 1. 规划完整路径
        path, directions, total_hops = src_router.full_path_route(msg.dst)
        logger.info(f"Time {self.env.now:.2f}: Wormhole routing path from Router{src_router_id} to Router{msg.dst}: {path}")
        
        # 2. 获取路径上的所有链路
        path_links = []
        for i in range(len(path) - 1):
            try:
                link = self.get_link_between_routers(path[i], path[i + 1])
                path_links.append(link)
            except ValueError as e:
                logger.error(f"Error getting link between Router{path[i]} and Router{path[i + 1]}: {e}")
                raise
        
        # 3. 预先请求所有链路资源
        link_requests = []
        logger.info(f"Time {self.env.now:.2f}: Requesting {len(path_links)} links for wormhole routing")
        
        for i, link in enumerate(path_links):
            req = link.linkentry.request()
            link_requests.append(req)
            logger.debug(f"Time {self.env.now:.2f}: Added link {i} (Router{path[i]} -> Router{path[i+1]}) to request batch.")
        
        # 4. 等待所有链路资源都可用 (All-or-Nothing)
        logger.info(f"Time {self.env.now:.2f}: Waiting for all {len(link_requests)} links to be available simultaneously...")
        yield self.env.all_of(link_requests)
        
        logger.info(f"Time {self.env.now:.2f}: All links acquired for wormhole routing")
        
        # 5. 计算整条路径的传输延迟（基于最窄带宽）
        min_bandwidth = min(link.width for link in path_links)
        
        if msg.msg_type == MsgType.DATA and msg.data:
            slice = Slice(tensor_slice=msg.data.tensor_slice)
            if msg.ins.data_type == DataType.FEAT:
                transmission_time = ceil(slice.size() * msg.ins.feat_precision, min_bandwidth)
            else:
                transmission_time = ceil(slice.size() * msg.ins.para_precision, min_bandwidth)
        else: # for MEM_REQUEST or other types
            transmission_time = 1 # Assume a small, fixed transmission time for request packets

        # 基础延迟是所有链路延迟之和
        total_delay = sum(link.delay * link.delay_factor for link in path_links)
        total_latency = total_delay + transmission_time
        
        logger.info(f"Time {self.env.now:.2f}: Wormhole transmission starting, latency={total_latency:.2f} (delay={total_delay:.2f}, transmission={transmission_time:.2f})")
        
        # 6. 记录开始时间并执行整条路径的传输
        msg.ins.record.ready_run_time.append(self.env.now)
        yield self.env.timeout(total_latency)
        
        # 7. 将消息传递给路径上所有相关的核心（不包括源）
        log_payload = f"data {msg.data.index}" if msg.data else f"{msg.msg_type.name}"
        logger.info(f"Time {self.env.now:.2f}: Wormhole transmission for {log_payload} arrived. Delivering to path destinations.")
        
        delivered_to = []
        # 遍历路径上的所有中间节点和终点 (path[0]是源，跳过)
        for router_id in path[1:]:
            if msg.should_deliver_to_core(router_id):
                target_router = self.routers[router_id]
                if hasattr(target_router, 'core_in') and target_router.core_in:
                    # 广播/多播时，所有接收方获取对同一消息对象的引用
                    target_router.core_out.store.put(msg)
                    delivered_to.append(router_id)
                else:
                    logger.warning(f"Time {self.env.now:.2f}: Wormhole path destination Router{router_id} has no core_in, message could not be delivered.")

        if delivered_to:
            logger.info(f"Time {self.env.now:.2f}: Message delivered via wormhole to Cores: {delivered_to}")
        
        # 8. 释放所有链路资源
        for i, req in enumerate(link_requests):
            path_links[i].linkentry.release(req)
            logger.debug(f"Time {self.env.now:.2f}: Released link {i}")
        
        logger.info(f"Time {self.env.now:.2f}: Wormhole transmission completed from Router{src_router_id} to Router{msg.dst}")

        return self