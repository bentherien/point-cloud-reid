import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import copy
import numpy as np
from functools import reduce



def plot_metrics_1d(data,
                    font,
                    font_legend,
                    title,
                    xlabel,
                    ylabel,
                    xticks,
                    xlabs,
                    ylim=(0,1.05,),
                    include_str=['f1','acc'],
                    filename=None,
                    label_dict=dict(),
                    figsize=(8,5),
                    key_iter=None,
                    use_interval=False,
                    pos_obs_num=None,
                    neg_obs_num=None,
                    marker_dict=dict(),
                    color_dict=dict(),
                    linestyle_dict = dict(),
                    xlab_font_size=10,
                    ylab_font_size=8,
                    ir=None,
                    include_str_map=None):
    
    plt.figure(figsize=figsize)
    
    
    if key_iter is None:
        
        if not pos_obs_num:
            pos_obs_num = np.array([data[x]['num_observations_pos'] for x in data])
            
        if not neg_obs_num:
            neg_obs_num = np.array([data[x]['num_observations_neg'] for x in data])
        iter_ = list(range(len(data)))
    else:
        
        
        if not pos_obs_num:
            pos_obs_num = np.array([data[x]['num_observations_pos'] for x in key_iter])
        if not neg_obs_num:
            neg_obs_num = np.array([data[x]['num_observations_neg'] for x in key_iter])
        
        iter_ = list(range(len(key_iter)))
        

        
        
    for i in list(data.values())[0].keys():
        if reduce(lambda a,b: a and b,[x not in i for x in include_str]):
            continue
        
        if key_iter is None:
            y = np.array([x.get(i,-1) for x in data.values()])
        else:
            y = np.array([data[x][i] for x in key_iter])
        
        if 'neg' in i:
            filter_ = np.where(np.logical_and(y != -1,neg_obs_num != 0))
        else:
            filter_ = np.where(np.logical_and(y != -1,pos_obs_num != 0))

        print(i)
        i_temp = i.split('--')[0]
        temp_lab = label_dict.get(i_temp,i_temp)
        if include_str_map is not None:
            ttemp = [x for x in include_str if x in i]
            temp_lab = include_str_map.get(ttemp[0],'') + temp_lab
        plt.plot(xticks[filter_] if ir is None else xticks[filter_][ir[0]:ir[1]], 
                (y[filter_] * 100) if ir is None else (y[filter_] * 100)[ir[0]:ir[1]], 
                 label=temp_lab,
                 linewidth=1.2, 
                 marker=marker_dict.get(temp_lab,'D'),
                 color=color_dict.get(temp_lab,None),
                 markersize=4,
                 linestyle=linestyle_dict.get(temp_lab,'--'))

    #all distances 
    plt.grid(linestyle='-')
    plt.legend(prop=font_legend)
    plt.title(title, fontdict=font)

    f = copy.deepcopy(font)
    print('ylab_font_size',ylab_font_size)
    f.update({'size':ylab_font_size})
    plt.ylabel(ylabel, fontdict=f)
    print('xlab_font_size:',xlab_font_size)

    f = copy.deepcopy(font)
    f.update({'size':xlab_font_size})
    plt.xlabel(xlabel, fontdict=f)

    if ylim is not None: 
        plt.ylim(*ylim)  
    plt.xscale('log',base=2)

    labels__ = ["{}\n{}\n{}".format(xlabs[x],pos_obs_num[x],neg_obs_num[x]) for x in iter_]
    plt.xticks(ticks=xticks if ir is None else xticks[ir[0]:ir[1]],
               labels=labels__ if ir is None else labels__[ir[0]:ir[1]],
               fontsize=8)
    
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
        
    plt.show()












class JSONToPlots(object):
    def __init__(self, 
                 data, 
                 key_map=dict(),
                 path='/mnt/bnas',
                 legend_font_size=10,
                 xlab_font_size=10,
                 ylab_font_size=8,
                 figsize=(12,5)):
        """
        data: dict structured as follows:
            model_name: 
                at_least_both:
                    f1
                    recall
                    precision
                    ...
                at_least_one:
                    ...
                for_a_pair:
                    ...
            ...

        """

        self.font = {'family': 'serif',
                     'color':  'black',
                     'weight': 'normal',
                     'size': 12,}
        self.font_legend = copy.deepcopy(self.font)
        self.font_legend['size'] = legend_font_size
        del self.font_legend['color']
        self.key_map = key_map

        self.path = path
        self.figsize = figsize

        self.xlab_font_size = xlab_font_size
        self.ylab_font_size = ylab_font_size

        self.data_pts = self.merge_data({k:v['results_per_points'] for k,v in data.items()})
        self.data_rgb = self.merge_data({k:v['results_per_visibility'] for k,v in data.items()})
        self.data_distance = self.merge_data({k:v['results_per_distance'] for k,v in data.items()})


        self.data_classes = dict()

        for k in list(data.values())[0].keys():
            if k not in ['results_per_points','results_per_visibility','results_per_distance'] and not k.startswith('FP_'):
                self.data_classes[k] = self.merge_data({model:v[k]['results_per_points'] for model,v in data.items()})


    def merge_data(self,data):
        temp = {}
        for model_name,dicts in data.items(): #iter over models
            model_name = self.key_map.get(model_name,model_name)
            for k,v in dicts.items(): #iter over at_least_both, at_least_one, for_a_pair
                temp[k] = temp.get(k,{})
                for k2,v2 in v.items(): #iter over num_points, visibility
                    temp[k][k2] = temp[k].get(k2,{})
                    for metric_name,metric_value in v2.items(): #iter over metrics
                        metric_name = self.key_map.get(metric_name,metric_name)
                        if 'num_observations' in metric_name:
                            temp[k][k2][metric_name] = metric_value
                        else:
                            temp[k][k2][model_name + '--' + metric_name] = metric_value

        return temp




    def get_results_num_points_points(self,data,include_str=['acc'],use_str=True,save=False,prefix='',linestyle_dict=dict(),
                                      marker_dict=dict(),label_dict=dict(),color_dict=dict(),ir=None,use_densities=False,include_str_map=None):
        font = self.font 
        font_legend = self.font_legend

        at_least_both = data['at_least_both']
        at_least_one = data['at_least_one']
        for_a_pair = data['for_a_pair']

        if use_densities:
            idx = [(x,x+1) for x in range(len(at_least_one))][1:]
            for x in idx:
                print(x)
                key_iter = [tuple(sorted((x,i),key=lambda x:x[0])) for i in idx]
                
                if use_str:
                    key_iter = [str(i) for i in key_iter]
                    
                if save:
                    filename = '{}/{}_acc_pts_{}_{}.pdf'.format(self.path,prefix,2**x[0],2**x[1])
                else:
                    filename = None

                plot_metrics_1d(data=for_a_pair,
                                font=font,
                                font_legend=font_legend,
                                title='Match f1-score and accuracy for #points$\in[2^{{{}}},2^{{{}}})$ '.format(
                                    x[0],x[1]),
                                xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                                ylabel='Accuracy (%) for pairs of objects with $[2^{{{}}},2^{{{}}})$ and x points'.format(x[0],x[1]),
                                ylim=None,#(0,1.05,),
                                include_str=include_str,
                                xticks=np.array([2**x for x in range(1,len(idx)+1)]),
                                xlabs=["$[2^{{{}}},2^{{{}}})$".format(x,(x+1)) for x in range(1,len(idx)+1)],
                                filename=filename,
                                label_dict=label_dict,
                                marker_dict=marker_dict,
                                linestyle_dict=linestyle_dict,
                                color_dict=color_dict,
                                figsize=self.figsize,
                                key_iter=key_iter,
                                use_interval=True,
                                xlab_font_size=self.xlab_font_size,
                                ylab_font_size=self.ylab_font_size,
                                ir=ir)
            
        
        if save:
            filename = '{}/{}_pts_acc_at_least_both.pdf'.format(self.path,prefix)
        else:
            filename = None
            
        plot_metrics_1d(data=at_least_both,
                font=font,
                font_legend=font_legend,
                title='',#'Match f1-score and accuracy for different #points',
                xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                ylabel="Accuracy (%) for matching two objects with at least #points each",
                #'Accuracy (%) for both objects w/ $\geq$ x #points',
                ylim=None,#(0,1.05,),
                include_str=include_str,
                xticks=np.array([2**x for x in range(len(at_least_both))]),
                xlabs=[2**x for x in range(len(at_least_both))],
                filename=filename,
                label_dict=label_dict,
                marker_dict=marker_dict,
                linestyle_dict=linestyle_dict,
                color_dict=color_dict,
                xlab_font_size=self.xlab_font_size,
                ylab_font_size=self.ylab_font_size,
                    include_str_map=include_str_map,
                figsize=self.figsize,
                        ir=ir)


        if save:
            filename = '{}/{}_pts_acc_at_least_one.pdf'.format(self.path,prefix)
        else:
            filename = None

        plot_metrics_1d(data=at_least_one,
                    font=font,
                    font_legend=font_legend,
                    title='',#'Match f1-score and accuracy for different #points',
                    xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                    ylabel="Accuracy (%) for matching two objects with one with at least #points",
                    # ylabel='Accuracy (%) for at least one object w/ $\geq$ x #points',
                    ylim=None,#(0,1.05,),
                    include_str=include_str,
                    xticks=np.array([2**x for x in range(len(at_least_one))]),
                    xlabs=[2**x for x in range(len(at_least_both))],
                    filename=filename,
                    label_dict=label_dict,
                    marker_dict=marker_dict,
                    linestyle_dict=linestyle_dict,
                    color_dict=color_dict,
                    xlab_font_size=self.xlab_font_size,
                    ylab_font_size=self.ylab_font_size,
                    figsize=self.figsize,
                    include_str_map=include_str_map,
                        ir=ir)


        
        
    def get_results_num_points_image(self,data,use_str=True,
                                      marker_dict=dict(),label_dict=dict(),color_dict=dict()):
        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 12,}

        font_legend = copy.deepcopy(font)
        font_legend['size'] = 8
        del font_legend['color']
        
        at_least_both = data['at_least_both']
        at_least_one = data['at_least_one']
        for_a_pair = data['for_a_pair']

        idx = [(x,x+1) for x in range(len(at_least_one))]
        for x in idx:
            print(x)
            key_iter = [tuple(sorted((x,i),key=lambda x:x[0])) for i in idx]
            
            if use_str:
                key_iter = [str(i) for i in key_iter]
                
                
            plot_metrics_1d(data=for_a_pair,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for #points$\in[2^{{{}}},2^{{{}}})$ '.format(
                            x[0],x[1]),
                        xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for objects w/ $\geq$  x #points',
                        ylim=None,#(0,1.05,),
                        include_str=['acc','f1'],
                        xticks=np.array([2**x for x in range(len(idx))]),
                        xlabs=["$[2^{{{}}},2^{{{}}})$".format(x,(x+1)) for x in range(len(idx))],
                        filename='/mnt/bnas/overleaf/deit_tiny_f1_acc_num_points_{}_{}.pdf'.format(2**x[0],2**x[1]),
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(12,5),
                        key_iter=key_iter,
                        use_interval=True,
                        ir=ir)
            
            
            
        plot_metrics_1d(data=at_least_both,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for different #points',
                        xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for both objects w/ $\geq$ x #points',
                        ylim=None,#(0,1.05,),
                        include_str=['f1','acc'],
                        xticks=np.array([2**x for x in range(len(at_least_both))]),
                        xlabs=[2**x for x in range(len(at_least_both))],
                        filename='/mnt/bnas/overleaf/deit_tiny_f1_acc_points_at_least_both.pdf',
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(8,5),
                        ir=ir)

        plot_metrics_1d(data=at_least_one,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for different #points',
                        xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for at least one object w/ $\geq$ x #points',
                        ylim=None,#(0,1.05,),
                        include_str=['f1','acc'],
                        xticks=np.array([2**x for x in range(len(at_least_one))]),
                        xlabs=[2**x for x in range(len(at_least_both))],
                        filename='/mnt/bnas/overleaf/deit_tiny_f1_acc_points_at_least_one.pdf',
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(8,5),
                        ir=ir)




    def get_results_visibility_points(self,data,use_str=True,
                                      marker_dict=dict(),label_dict=dict(),color_dict=dict(),ir=None):
        at_least_both = data['at_least_both']
        at_least_one = data['at_least_one']
        for_a_pair = data['for_a_pair']
        
        plot_metrics_1d(data=at_least_both,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for objects with varying visibility',
                        xlabel='%visibility \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for both objects w/ $\geq$ x %visibility',
                        ylim=None,#(0,1.05,),
                        include_str=['f1','acc'],
                        xticks=np.array([x+1 for x in range(len(at_least_one))]),
                        xlabs=['40%','60%','80%','100%'],
                        filename='/mnt/bnas/overleaf/point_trans_at_least_both_visibility_image.pdf',
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(8,5),
                        ir=ir)

        plot_metrics_1d(data=at_least_one,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for objects with varying visibility',
                        xlabel='%visibility \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for at least one objects w/ $\geq$ x %visibility',
                        ylim=None,#(0,1.05,),
                        xticks=np.array([x+1 for x in range(len(at_least_one))]),
                        xlabs=['40%','60%','80%','100%'],
                        include_str=['f1','acc'],
                        filename='/mnt/bnas/overleaf/point_trans_at_least_one_visibility_image.pdf',
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(8,5),
                        ir=ir)
        


            # idx = [(x,x+1) for x in range(len(at_least_one))]
        vis_levels = ['[0%,40%)','[40%,60%)','[60%,80%)','[80%,100%)']

        for x in range(len(at_least_one)):
            key_iter=[tuple(sorted([x,i])) for i in range(len(at_least_one))]
            
            if use_str:
                key_iter = [str(i) for i in key_iter]
                pos_obs_num=np.array([for_a_pair[i]['num_observations_pos'] if i[0] < i[1] \
                                            else for_a_pair[str((i[1],i[0],))]['num_observations_pos']
                                                for i in key_iter])
                neg_obs_num=np.array([for_a_pair[i]['num_observations_neg'] if i[0] < i[1] \
                                else for_a_pair[str((i[1],i[0],))]['num_observations_neg']
                                    for i in key_iter])
                
                
            else:
                
                pos_obs_num=np.array([for_a_pair[i]['num_observations_pos'] if i[0] < i[1] \
                                    else for_a_pair[(i[1],i[0])]['num_observations_pos']
                                    for i in key_iter])
                neg_obs_num=np.array([for_a_pair[i]['num_observations_neg'] if i[0] < i[1] \
                                    else for_a_pair[(i[1],i[0])]['num_observations_neg']
                                    for i in key_iter])
            
                
                
            plot_metrics_1d(data=for_a_pair,
                            font=font,
                            font_legend=font_legend,
                            title='Match f1-score and accuracy for %visibility$\in${}'.format(
                                vis_levels[x]),
                            xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                            ylabel='score (%) for objects w/ $\geq$ x  %visibility',
                            ylim=None,#(0,1.05,),
                            include_str=['acc','f1'],
                            xticks=np.array([1,2,3,4]),
                            xlabs=vis_levels,
                            filename='/mnt/bnas/overleaf/deit_tiny_f1_acc_points_{}.pdf'.format(vis_levels[x]),
                            label_dict=label_dict,
                            marker_dict=marker_dict,
                            color_dict=color_dict,
                            figsize=(12,5),
                            key_iter=key_iter,
                            pos_obs_num=pos_obs_num,
                            neg_obs_num=neg_obs_num,
                            use_interval=True,
                        ir=ir)
            
            
            
            
    def get_results_visibility_image(self,data,use_str=True,
                                      marker_dict=dict(),label_dict=dict(),color_dict=dict(),ir=None):
        at_least_both = data['at_least_both']
        at_least_one = data['at_least_one']
        for_a_pair = data['for_a_pair']
        
        plot_metrics_1d(data=at_least_both,
                font=font,
                font_legend=font_legend,
                title='Match f1-score and accuracy for objects with varying visibility',
                xlabel='%visibility \nNumber of positive samples \nNumber of negative samples',
                ylabel='score (%) for both objects w/ $\geq$ x %visibility',
                ylim=None,#(0,1.05,),
                include_str=['f1','acc'],
                xticks=np.array([x+1 for x in range(len(at_least_one))]),
                xlabs=['40%','60%','80%','100%'],
                filename='/mnt/bnas/overleaf/at_least_both_visibility_image.pdf',
                label_dict=label_dict,
                marker_dict=marker_dict,
                color_dict=color_dict,
                figsize=(8,5),
                        ir=ir)

        plot_metrics_1d(data=at_least_one,
                    font=font,
                    font_legend=font_legend,
                    title='Match f1-score and accuracy for objects with varying visibility',
                    xlabel='%visibility \nNumber of positive samples \nNumber of negative samples',
                    ylabel='score (%) for at least one objects w/ $\geq$ x %visibility',
                    ylim=None,#(0,1.05,),
                    xticks=np.array([x+1 for x in range(len(at_least_one))]),
                    xlabs=['40%','60%','80%','100%'],
                    include_str=['f1','acc'],
                    filename='/mnt/bnas/overleaf/at_least_one_visibility_image.pdf',
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                    figsize=(8,5),
                        ir=ir)
        


            # idx = [(x,x+1) for x in range(len(at_least_one))]
        vis_levels = ['[0%,40%)','[40%,60%)','[60%,80%)','[80%,100%)']

        for x in range(len(at_least_one)):
            key_iter=[tuple(sorted([x,i])) for i in range(len(at_least_one))]
            
            if use_str:
                key_iter = [str(i) for i in key_iter]
                pos_obs_num=np.array([for_a_pair[i]['num_observations_pos'] if i[0] < i[1] \
                                            else for_a_pair[str((i[1],i[0],))]['num_observations_pos']
                                                for i in key_iter])
                neg_obs_num=np.array([for_a_pair[i]['num_observations_neg'] if i[0] < i[1] \
                                else for_a_pair[str((i[1],i[0],))]['num_observations_neg']
                                    for i in key_iter])
                
                
            else:
                
                pos_obs_num=np.array([for_a_pair[i]['num_observations_pos'] if i[0] < i[1] \
                                    else for_a_pair[(i[1],i[0])]['num_observations_pos']
                                    for i in key_iter])
                neg_obs_num=np.array([for_a_pair[i]['num_observations_neg'] if i[0] < i[1] \
                                    else for_a_pair[(i[1],i[0])]['num_observations_neg']
                                    for i in key_iter])
            
                
                
            plot_metrics_1d(data=for_a_pair,
                        font=font,
                        font_legend=font_legend,
                        title='Match f1-score and accuracy for %visibility$\in${}'.format(
                            vis_levels[x]),
                        xlabel='#points \nNumber of positive samples \nNumber of negative samples',
                        ylabel='score (%) for objects w/ $\geq$ x  %visibility',
                        ylim=None,#(0,1.05,),
                        include_str=['acc','f1'],
                        xticks=np.array([1,2,3,4]),
                        xlabs=vis_levels,
                        filename='/mnt/bnas/overleaf/deit_tiny_f1_acc_points_{}.pdf'.format(vis_levels[x]),
                        label_dict=label_dict,
                        marker_dict=marker_dict,
                        color_dict=color_dict,
                        figsize=(12,5),
                        key_iter=key_iter,
                        pos_obs_num=pos_obs_num,
                        neg_obs_num=neg_obs_num,
                        use_interval=True,
                        ir=ir)