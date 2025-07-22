import simpy
import logging
import contextlib
from enum import IntEnum
from src.arch_config import LinkConfig, RouterConfig, NoCConfig
from src.sim_type import Data, Message, ceil, Slice, Direction, DataType
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
    def __init__(self, env, config):
        self.env = env
        self.width = config.width
        self.delay = config.delay
        # 按数据的index排序，需要Message的lt方法
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
                yield self.env.process(self.route(msg, next_dir, next_router))

    def route(self, msg: Message, next_dir, next_router):
        match next_dir:
            case Direction.NORTH:
                yield self.north_out.put(msg)
            case Direction.SOUTH:
                yield self.south_out.put(msg)
            case Direction.EAST:
                yield self.east_out.put(msg)
            case Direction.WEST:
                yield self.west_out.put(msg)

        logger.info(f"Time {self.env.now:.2f}: Router{self.id} finish sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")

    def route_core(self, msg):
        yield self.core_out.put(msg)
        logger.info(f"Time {self.env.now:.2f}: Finish putting data{msg.data.index} to Core{self.id}")
 
    def run(self):
        while True:
            all_possible_channels = [(self.north_in, 0), (self.south_in, 1), (self.east_in, 2), (self.west_in, 3), (self.core_in, 4)]
            #有的没有四个方向
            all_channels = [channel for channel in all_possible_channels if channel[0] is not None]

            # with self.north_in.get() as n, self.south_in.get() as s, self.east_in.get() as e, self.west_in.get() as w, self.core_in.get() as c:
            with contextlib.ExitStack() as stack:
                # all_in_channels = [self.north_in.get(), self.south_in.get(), self.east_in.get(), self.west_in.get(), self.core_in.get()]
                all_events = [stack.enter_context(channel[0].get()) for channel in all_channels]

                events = self.env.any_of(all_events)
                #等到至少有一个触发
                result = yield events

                for id, event in enumerate(all_events):
                    
                    if event.triggered:
                        msg = event.value

                        # 检查是否应该传递给本地核心（支持路径广播）
                    
                        if msg.should_deliver_to_core(self.id):
                            logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id} (broadcast delivery).")
                            self.env.process(self.route_core(msg))
                        
                        # 如果不是最终目标，继续路由
                        if msg.dst != self.id:
                            next_dir, next_router = self.calculate_next_router(msg.dst)
                            logger.info(f"Time {self.env.now:.2f}: Router{self.id} start sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")
                            self.env.process(self.route(msg, next_dir, next_router))
                        
                        channel = None
                        match all_channels[id][1]:
                            case 0: channel = self.north_in
                            case 1: channel = self.south_in
                            case 2: channel = self.east_in
                            case 3: channel = self.west_in
                            case 4: channel = self.core_in
                        
                        while channel.len() > 0:
                            msg = yield channel.get()

                            # 检查是否应该传递给本地核心（支持路径广播）
                            if msg.should_deliver_to_core(self.id):
                                logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id} (broadcast delivery).")
                                self.env.process(self.route_core(msg))
                            
                            # 如果不是最终目标，继续路由
                            if msg.dst != self.id:
                                next_dir, next_router = self.calculate_next_router(msg.dst)
                                logger.info(f"Time {self.env.now:.2f}: Router{self.id} finished sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")
                                self.env.process(self.route(msg, next_dir, next_router))

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
        
        logger.info(f"Time {self.env.now:.2f}: Router{self.id} initiating wormhole routing for data{msg.data.index} to Router{msg.dst}")
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
            logger.debug(f"Time {self.env.now:.2f}: Requesting link {i} (Router{path[i]} -> Router{path[i+1]})")
        
        # 4. 等待所有链路资源都可用
        for i, req in enumerate(link_requests):
            yield req
            logger.debug(f"Time {self.env.now:.2f}: Acquired link {i} for wormhole routing")
        
        logger.info(f"Time {self.env.now:.2f}: All links acquired for wormhole routing")
        
        # 5. 计算整条路径的传输延迟（基于最窄带宽）
        min_bandwidth = min(link.width for link in path_links)
        
        slice = Slice(tensor_slice=msg.data.tensor_slice)
        if msg.ins.data_type == DataType.FEAT:
            transmission_time = ceil(slice.size() * msg.feat_precision, min_bandwidth)
        else:
            transmission_time = ceil(slice.size() * msg.para_precision, min_bandwidth)
        
        # 基础延迟是所有链路延迟之和
        total_delay = sum(link.delay * link.delay_factor for link in path_links)
        total_latency = total_delay + transmission_time
        
        logger.info(f"Time {self.env.now:.2f}: Wormhole transmission starting, latency={total_latency:.2f} (delay={total_delay:.2f}, transmission={transmission_time:.2f})")
        
        # 6. 记录开始时间并执行整条路径的传输
        msg.ins.record.ready_run_time.append(self.env.now)
        yield self.env.timeout(total_latency)
        
        # 7. 将消息放入目标Router的输入缓冲区
        target_router = self.routers[msg.dst]
        if hasattr(target_router, 'core_in') and target_router.core_in:
            target_router.core_in.store.put(msg)
            logger.info(f"Time {self.env.now:.2f}: Message delivered to Router{msg.dst} core input")
        else:
            logger.warning(f"Time {self.env.now:.2f}: Router{msg.dst} has no core_in, message may be lost")
        
        # 8. 释放所有链路资源
        for i, req in enumerate(link_requests):
            path_links[i].linkentry.release(req)
            logger.debug(f"Time {self.env.now:.2f}: Released link {i}")
        
        logger.info(f"Time {self.env.now:.2f}: Wormhole transmission completed from Router{src_router_id} to Router{msg.dst}")

        return self