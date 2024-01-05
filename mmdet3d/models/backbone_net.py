"""
code below this line is taken/modified from https://github.com/fpthink/STNet/blob/main/modules/pointnet2_utils.py
"""
import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstractionEdgeSA, PointNetFeaturePropagationSA


class Pointnet_Backbone(nn.Module):
    """
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

class Pointnet_Backbone(nn.Module):
    """
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True, conv_out=32, mul=1, radius=[0.3,0.5,0.7], nsample=[32,48,48]):
        super(Pointnet_Backbone, self).__init__()
        
        mul = mul
        sa1 = 32 * mul
        sa2 = 64 * mul
        sa3 = 128 * mul

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[0],
                nsample=nsample[0],
                mlp=[input_channels, sa1, sa1, sa1],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[1],
                nsample=nsample[1],
                mlp=[sa2, sa2, sa2, sa2],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[2],
                nsample=nsample[2],
                mlp=[sa3, sa3, sa3, sa3],
                sampling="RANDOM",
                use_xyz=use_xyz,
                use_knn=True
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[67,sa1,sa1], mlp_inte=[sa2, 3, sa2, sa2, sa1]))
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[160,sa3,sa2], mlp_inte=[sa3, sa1, sa3, sa2, sa2]))    # 160=128+32
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[192,sa3,sa3], mlp_inte=[sa3, sa2, sa3, sa2, sa3]))  # 192=128+64


        self.cov_final = nn.Conv1d(sa1, conv_out, kernel_size=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)   # [B, N, 3], [B, C, N]
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
#             print('SA in:',i, l_xyz[i].shape, l_features[i].shape if l_features[i] is not None else None, numpoints[i])
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
#             print('SA out:',i, li_xyz.shape, li_features.shape)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        l_features[0] = xyz.transpose(1, 2).contiguous()
        for i in [2, 1, 0]:
#             print('FP IN',i, l_xyz[i].shape, l_xyz[i+1].shape, l_features[i].shape, l_features[i+1].shape)
            l_features[i] = self.FP_modules[i](l_xyz[i], l_xyz[i+1], l_features[i], l_features[i+1])
#             print('FP OUT:',i, l_features[i].shape)

        return l_xyz[0], self.cov_final(l_features[0])  # [B, N, 3], [B, 32, N]