# Functions/PostProcessing/Plotting.py

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             roc_auc_score, average_precision_score,
                             brier_score_loss)
from scipy.io import savemat

from Functions.Imports import *

_Binary_Cavity_Key_Aliases = {"cavity_consumed", "consumed_cavity"}


def _Is_Binary_Cavity_Keys(Output_Keys: list) -> bool:
    """True if Output_Keys is the single binary cavity-mask output."""
    if len(Output_Keys) != 1:
        return False
    return Output_Keys[0].strip().lower() in _Binary_Cavity_Key_Aliases


def PostProcessPlotting(Epoch_Tensor, Loss_Tensor, IoU_Tensor, Fscore_Tensor, Accuracy_Tensor,
                        Loss_Type, Result_Path, Model_Name, n_epoch):
    """Loss, IoU, FScore and Accuracy over the run, for all three splits."""
    Epoch_Tensor    = Epoch_Tensor.cpu().numpy()
    Loss_Tensor     = Loss_Tensor.cpu().numpy()
    IoU_Tensor      = IoU_Tensor.cpu().numpy()
    Fscore_Tensor   = Fscore_Tensor.cpu().numpy()
    Accuracy_Tensor = Accuracy_Tensor.cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    Panels = [(axs[0, 0], Loss_Tensor,     f'Loss ({Loss_Type})', None),
              (axs[0, 1], IoU_Tensor,      'IoU',                 (0, 1)),
              (axs[1, 0], Fscore_Tensor,   'F-Score',             (0, 1)),
              (axs[1, 1], Accuracy_Tensor, 'Accuracy',            (0, 1))]

    Styles = [('r', '-'), ('b', '--'), ('c', '-.')]   # Train, Valid, Test

    for Axis, Tensor, Label, Y_Limits in Panels:
        for Column, (Colour, Line_Style) in enumerate(Styles):
            Axis.plot(Epoch_Tensor, Tensor[:, Column], color=Colour, linestyle=Line_Style)

        Axis.set_xlabel('Epoch')
        Axis.set_ylabel(Label)
        Axis.grid(True, which='both', axis='both')
        Axis.set_xlim(0, n_epoch)
        if Y_Limits is not None:
            Axis.set_ylim(*Y_Limits)

    fig.legend(labels=['Training', 'Validation', 'Testing'], loc='lower center', ncol=3, frameon=True)

    plt.savefig(os.path.join(Result_Path, f'{Model_Name}_Training_Metrics.png'), dpi=600)
    plt.close(fig)

    return


def Plot_Single_Dataset(args):
    """One (sample, time step) of the binary cavity output: four isometric voxel
    views, plus mid-slice probability fields when Ensemble_Toggle is set. The
    voxel fields are also written to a .mat alongside the figures."""
    (dataset, Time_Index, X, Y, Z,
     true_mask, pred_mask, pred_score, backfill,
     IoU, FScore, Accuracy, Loss, GrowthIoU,
     Result_Path, Ensemble_Toggle) = args

    views = [(35, 45,  "Northeast Isometric View"),
             (35, 225, "Southwest Isometric View")]

    # Pred_Score -> Continuous for Colormap
    true_mask     = numpy.squeeze(true_mask).astype(bool)
    pred_mask     = numpy.squeeze(pred_mask).astype(bool)
    pred_score    = numpy.squeeze(pred_score).astype(float)
    backfill_mask = numpy.round(numpy.squeeze(backfill).astype(float)).astype(bool)

    # RGBA
    true_color     = (0.12, 0.47, 0.71, 0.45)  # blue
    pred_color     = (0.90, 0.30, 0.10, 0.45)  # red
    backfill_color = (0.20, 0.70, 0.20, 0.45)  # green

    facecolors_true = numpy.zeros(true_mask.shape + (4,), dtype=float)
    facecolors_pred = numpy.zeros(pred_mask.shape + (4,), dtype=float)
    facecolors_back = numpy.zeros(backfill_mask.shape + (4,), dtype=float)

    facecolors_true[true_mask]     = true_color
    facecolors_pred[pred_mask]     = pred_color
    facecolors_back[backfill_mask] = backfill_color

    nx, ny, nz = pred_mask.shape
    x_labels = numpy.linspace(numpy.min(X), numpy.max(X), 5).astype(int)
    y_labels = numpy.linspace(numpy.min(Y), numpy.max(Y), 5).astype(int)
    z_labels = numpy.linspace(numpy.min(Z), numpy.max(Z), 5).astype(int)

    Sample_Dir = os.path.join(Result_Path, f"{dataset}", f"T{Time_Index + 1}")
    os.makedirs(Sample_Dir, exist_ok=True)

    Mat_Fields = {'True_Mask':  true_mask.astype(numpy.uint8),
                  'Pred_Mask':  pred_mask.astype(numpy.uint8),
                  'Pred_Score': pred_score.astype(numpy.float32),
                  'Backfill':   backfill_mask.astype(numpy.uint8),
                  'X': numpy.asarray(X, dtype=numpy.float32),
                  'Y': numpy.asarray(Y, dtype=numpy.float32),
                  'Z': numpy.asarray(Z, dtype=numpy.float32),
                  'Dataset':    str(dataset),
                  'Time_Index': int(Time_Index + 1),
                  'IoU': float(IoU), 'FScore': float(FScore),
                  'Accuracy': float(Accuracy), 'Loss': float(Loss)}
    if GrowthIoU is not None:
        Mat_Fields['GrowthIoU'] = float(GrowthIoU)

    savemat(os.path.join(Sample_Dir, f'{dataset}_T{Time_Index + 1}_Fields.mat'),
            Mat_Fields, do_compression=True)

    legend_elements = [
        Patch(facecolor=true_color, edgecolor='k', label='Ground Truth Consumed Blocks'),
        Patch(facecolor=pred_color, edgecolor='k', label='Predicted Consumed Blocks'),
        Patch(facecolor=backfill_color, edgecolor='k', label='Backfilled Blocks')
    ]

    def setup_ax(ax):
        ax.set_proj_type('ortho')
        ax.set_box_aspect((nx, ny, nz))
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)

        ax.set_xticks(numpy.linspace(0, nx, 5))
        ax.set_yticks(numpy.linspace(0, ny, 5))
        ax.set_zticks(numpy.linspace(0, nz, 5))

        ax.set_xticklabels(x_labels, rotation=30)
        ax.set_yticklabels(y_labels, rotation=30)
        ax.set_zticklabels(z_labels, rotation=0)

        ax.set_xlabel("X", labelpad=8)
        ax.set_ylabel("Y", labelpad=8)
        ax.set_zlabel("Z", labelpad=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 2D Mid-Slice Probability Fields
    if Ensemble_Toggle is True:
        ix, iy, iz = nx // 2, ny // 2, nz // 2

        x_min, x_max = numpy.min(X), numpy.max(X)
        y_min, y_max = numpy.min(Y), numpy.max(Y)
        z_min, z_max = numpy.min(Z), numpy.max(Z)

        slices = [
            ("XY Slice (Z Mid)", pred_score[:, :, iz], true_mask[:, :, iz],
             [x_min, x_max, y_min, y_max], "X", "Y"),
            ("XZ Slice (Y Mid)", pred_score[:, iy, :], true_mask[:, iy, :],
             [x_min, x_max, z_min, z_max], "X", "Z"),
            ("YZ Slice (X Mid)", pred_score[ix, :, :], true_mask[ix, :, :],
             [y_min, y_max, z_min, z_max], "Y", "Z"),
        ]

        Contour_Levels = [(0.5,  "red",    1.5, "Prediction Boundary (Score > 0.5)"),
                          (0.3,  "cyan",   1.0, "Prediction Boundary (Score > 0.3)"),
                          (0.1,  "yellow", 1.0, "Prediction Boundary (Score > 0.1)"),
                          (0.01, "green",  1.0, "Prediction Boundary (Score > 1e-2)")]

        fig2 = plt.figure(figsize=(18, 5), constrained_layout=True)
        gs = gridspec.GridSpec(1, 6, figure=fig2, width_ratios=[1.0, 0.05, 1.0, 0.05, 1.0, 0.05], wspace=0.35)
        ax2  = [fig2.add_subplot(gs[0, 0]), fig2.add_subplot(gs[0, 2]), fig2.add_subplot(gs[0, 4])]
        cax2 = [fig2.add_subplot(gs[0, 1]), fig2.add_subplot(gs[0, 3]), fig2.add_subplot(gs[0, 5])]

        legend_handles = [Line2D([0], [0], color=Colour, lw=Width, linestyle="-", label=Label)
                          for _, Colour, Width, Label in Contour_Levels]
        legend_handles.append(Line2D([0], [0], color="blue", lw=1.5, linestyle="--",
                                     label="Ground Truth Boundary"))

        for a, cax, (title2, img, gt, extent, xlabel, ylabel) in zip(ax2, cax2, slices):
            im = a.imshow(img.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=extent)

            for Level, Colour, Width, _ in Contour_Levels:
                a.contour((img > Level).T, levels=[0.5], linewidths=Width, colors=Colour,
                          linestyles="-", extent=extent)

            a.contour(gt.T.astype(float), levels=[0.5], linewidths=1.5, colors="blue",
                      linestyles="--", extent=extent)

            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)
            a.set_title(title2)

            cb = fig2.colorbar(im, cax=cax)
            cb.set_label("Predicted Probability")

        fig2.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.15))
        fig2.suptitle(f"{dataset} (T={Time_Index + 1}) Prediction Field Slices", y=1.08)
        plt.savefig(os.path.join(Sample_Dir, f"{dataset}_T{Time_Index + 1}_Prediction_Slices.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig2)

    # Build the metrics line for the title -- GrowthIoU only appended when
    # available (Incremental problems); omitted entirely for Static.
    Metrics_Line = f'IoU: {IoU:.4f}'
    if GrowthIoU is not None:
        Metrics_Line += f', GrowthIoU: {GrowthIoU:.4f}'
    Metrics_Line += f', FScore: {FScore:.4f}, Accuracy: {Accuracy:.4f}, Loss: {Loss:.4f}'

    # 3D voxel views
    for el, az, title in views:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        setup_ax(ax)

        ax.voxels(backfill_mask, facecolors=facecolors_back, edgecolor=None, linewidth=0.0)
        ax.voxels(true_mask, facecolors=facecolors_true, edgecolor=None, linewidth=0.0)
        ax.voxels(pred_mask, facecolors=facecolors_pred, edgecolor=None, linewidth=0.0)

        ax.view_init(elev=el, azim=az)
        ax.set_title(
            f'{dataset} (T={Time_Index + 1}) Cavern_Shape: {title}\n{Metrics_Line}',
            pad=14
        )

        fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True,
                   bbox_to_anchor=(0.5, 0.02))

        plt.savefig(os.path.join(Sample_Dir, f'{dataset}_T{Time_Index +1}_Cavern_Shape_{title}.png'),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return


def _Extract_Sample_Metric(Metric_Values, i: int, t: int) -> float:
    """Scalar metric for sample i at time t, from a (Nb,) or (Nb, Nt) array."""
    Values = numpy.asarray(Metric_Values)
    if Values.ndim >= 2:
        return float(Values[i, t])
    return float(Values[i])


def PlotPrediction(Output_Batch_Pred: torch.Tensor, Output_Batch: torch.Tensor, Dataset_Name_Batch: list,
                   Coord_Batch: torch.Tensor, Backfill_Batch: torch.Tensor, Results_Samples: dict,
                   Result_Path: str, Ensemble_Toggle: bool, Output_Keys: list,
                   Threshold: float = 0.5):
    """One plotting task per (sample, time step) from (Nb, Nx, Ny, Nz, Nt, Nc)
    batches, dispatched in parallel. Binary predictions cut at Threshold.

    Results_Samples['GrowthIoU'] is optional -- only present for Incremental
    problems. When absent, GrowthIoU is passed through as None per sample,
    and Plot_Single_Dataset omits it from the plot title entirely."""
    Nb = Output_Batch_Pred.size(0)
    Nt = Output_Batch_Pred.size(4)

    Is_Cavity_Consumed = _Is_Binary_Cavity_Keys(Output_Keys)

    IoU_Batch      = Results_Samples['IoU']
    Accuracy_Batch = Results_Samples['Accuracy']
    FScore_Batch   = Results_Samples['FScore']
    Loss_Batch     = Results_Samples['Loss']
    GrowthIoU_Batch = Results_Samples.get('GrowthIoU', None)  # Incremental only

    tasks = []
    for i in range(Nb):
        for t in range(Nt):

            X = Coord_Batch[i, :, :, :, t, 0].cpu().numpy()
            Y = Coord_Batch[i, :, :, :, t, 1].cpu().numpy()
            Z = Coord_Batch[i, :, :, :, t, 2].cpu().numpy()
            Backfill = Backfill_Batch[i, :, :, :, t].cpu().numpy()

            True_Output = Output_Batch[i, :, :, :, t, :].cpu().numpy()                # (nx,ny,nz,n_keys)
            Pred_Output = Output_Batch_Pred[i, :, :, :, t, :].detach().cpu().numpy()  # (nx,ny,nz,n_keys)

            IoU      = _Extract_Sample_Metric(IoU_Batch, i, t)
            FScore   = _Extract_Sample_Metric(FScore_Batch, i, t)
            Accuracy = _Extract_Sample_Metric(Accuracy_Batch, i, t)
            Loss     = _Extract_Sample_Metric(Loss_Batch, i, t)
            GrowthIoU = (_Extract_Sample_Metric(GrowthIoU_Batch, i, t)
                        if GrowthIoU_Batch is not None else None)

            if Is_Cavity_Consumed:
                True_Mask  = (True_Output == 1).squeeze(-1)
                Pred_Score = Pred_Output.squeeze(-1).astype(float)
                Pred_Mask  = (Pred_Score >= Threshold)

                tasks.append((
                    Dataset_Name_Batch[i], t, X, Y, Z,
                    True_Mask, Pred_Mask, Pred_Score, Backfill,
                    IoU, FScore, Accuracy, Loss, GrowthIoU,
                    Result_Path, Ensemble_Toggle
                ))
            else:
                tasks.append((
                    Dataset_Name_Batch[i], t, Output_Keys, X, Y, Z,
                    True_Output, Pred_Output,
                    Result_Path, Ensemble_Toggle
                ))

    if not tasks:
        return

    num_workers = min(mp.cpu_count(), len(tasks))
    ctx = mp.get_context("spawn")

    worker = Plot_Single_Dataset if Is_Cavity_Consumed else Plot_Single_Dataset_Keys

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        list(executor.map(worker, tasks))

    return


def Plot_Single_Dataset_Keys(args):
    """Generic counterpart to Plot_Single_Dataset for non-binary output keys.
    Ground Truth / Prediction / Absolute Error scatter, zero-valued voxels
    masked out per panel. Fields are also written to a .mat per key."""
    (dataset, Time_Index, output_keys, X, Y, Z,
     true_output, pred_output,
     Result_Path, Ensemble_Toggle) = args

    views = [(35, 45,  "Northeast Isometric View"),
             (35, 225, "Southwest Isometric View"),]

    true_output = numpy.asarray(true_output).astype(float)   # (nx, ny, nz, n_keys)
    pred_output = numpy.asarray(pred_output).astype(float)   # (nx, ny, nz, n_keys)

    nx, ny, nz = true_output.shape[:3]
    x_labels = numpy.linspace(numpy.min(X), numpy.max(X), 5).astype(int)
    y_labels = numpy.linspace(numpy.min(Y), numpy.max(Y), 5).astype(int)
    z_labels = numpy.linspace(numpy.min(Z), numpy.max(Z), 5).astype(int)

    xi, yi, zi = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny), numpy.arange(nz), indexing='ij')

    Sample_Dir = os.path.join(Result_Path, f"{dataset}", f"T{Time_Index + 1}")
    os.makedirs(Sample_Dir, exist_ok=True)

    Zero_Tol = 1e-6

    def setup_ax(ax):
        ax.set_proj_type('ortho')
        ax.set_box_aspect((nx, ny, nz))
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)

        ax.set_xticks(numpy.linspace(0, nx, 5))
        ax.set_yticks(numpy.linspace(0, ny, 5))
        ax.set_zticks(numpy.linspace(0, nz, 5))

        ax.set_xticklabels(x_labels, rotation=30)
        ax.set_yticklabels(y_labels, rotation=30)
        ax.set_zticklabels(z_labels, rotation=0)

        ax.set_xlabel("X", labelpad=8)
        ax.set_ylabel("Y", labelpad=8)
        ax.set_zlabel("Z", labelpad=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    for k, key in enumerate(output_keys):

        true_field  = true_output[..., k]
        pred_field  = pred_output[..., k]
        error_field = numpy.abs(pred_field - true_field)

        value_min = float(min(true_field.min(), pred_field.min()))
        value_max = float(max(true_field.max(), pred_field.max()))
        if value_min == value_max:
            value_max = value_min + 1e-6

        Error_Max = float(error_field.max()) + 1e-9

        rmse = float(numpy.sqrt(numpy.mean((pred_field - true_field) ** 2)))
        mae  = float(numpy.mean(error_field))

        True_Nonzero  = numpy.abs(true_field) > Zero_Tol
        Pred_Nonzero  = numpy.abs(pred_field) > Zero_Tol
        Error_Nonzero = error_field > Zero_Tol

        key_dir = os.path.join(Sample_Dir, key)
        os.makedirs(key_dir, exist_ok=True)

        savemat(os.path.join(key_dir, f'{dataset}_T{Time_Index + 1}_{key}_Fields.mat'),
                {'True_Field':  true_field.astype(numpy.float32),
                 'Pred_Field':  pred_field.astype(numpy.float32),
                 'Error_Field': error_field.astype(numpy.float32),
                 'X': numpy.asarray(X, dtype=numpy.float32),
                 'Y': numpy.asarray(Y, dtype=numpy.float32),
                 'Z': numpy.asarray(Z, dtype=numpy.float32),
                 'Dataset':    str(dataset),
                 'Time_Index': int(Time_Index + 1 ),
                 'Key':        str(key),
                 'RMSE': rmse, 'MAE': mae},
                do_compression=True)

        # 3D scatter views
        for el, az, title in views:
            fig, axes = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"},
                                     figsize=(18, 6))

            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.18, top=0.86, wspace=0.35)

            setup_ax(axes[0])
            if numpy.any(True_Nonzero):
                sc0 = axes[0].scatter(xi[True_Nonzero], yi[True_Nonzero], zi[True_Nonzero],
                                      c=true_field[True_Nonzero], cmap='viridis',
                                      vmin=value_min, vmax=value_max, marker='s', s=14, alpha=0.85)
            else:
                # empty scatter still needed so the shared colorbar has a mappable
                sc0 = axes[0].scatter([], [], [], c=[], cmap='viridis', vmin=value_min, vmax=value_max)
                axes[0].text2D(0.5, 0.5, "All Zero", transform=axes[0].transAxes, ha='center', va='center')
            axes[0].view_init(elev=el, azim=az)
            axes[0].set_title("Ground Truth", pad=6)

            setup_ax(axes[1])
            if numpy.any(Pred_Nonzero):
                axes[1].scatter(xi[Pred_Nonzero], yi[Pred_Nonzero], zi[Pred_Nonzero],
                                c=pred_field[Pred_Nonzero], cmap='viridis',
                                vmin=value_min, vmax=value_max, marker='s', s=14, alpha=0.85)
            else:
                axes[1].text2D(0.5, 0.5, "All Zero", transform=axes[1].transAxes, ha='center', va='center')
            axes[1].view_init(elev=el, azim=az)
            axes[1].set_title("Prediction", pad=6)

            setup_ax(axes[2])
            if numpy.any(Error_Nonzero):
                sc2 = axes[2].scatter(xi[Error_Nonzero], yi[Error_Nonzero], zi[Error_Nonzero],
                                      c=error_field[Error_Nonzero], cmap='hot',
                                      vmin=0, vmax=Error_Max, marker='s', s=14, alpha=0.85)
            else:
                sc2 = axes[2].scatter([], [], [], c=[], cmap='hot', vmin=0, vmax=Error_Max)
                axes[2].text2D(0.5, 0.5, "No Error", transform=axes[2].transAxes, ha='center', va='center')
            axes[2].view_init(elev=el, azim=az)
            axes[2].set_title("Absolute Error", pad=6)

            cb0 = fig.colorbar(sc0, ax=[axes[0], axes[1]], shrink=0.6, pad=0.08, location='bottom')
            cb0.set_label(key)
            cb2 = fig.colorbar(sc2, ax=axes[2], shrink=0.6, pad=0.08, location='bottom')
            cb2.set_label(f"{key} Absolute Error")

            fig.suptitle(f'{dataset} (T={Time_Index + 1}) {key}: {title}\n'
                         f'RMSE: {rmse:.4f}, MAE: {mae:.4f} (zero-valued voxels hidden)',
                         y=0.97)

            plt.savefig(os.path.join(key_dir, f'{dataset}_T{Time_Index + 1}_{key}_{title}.png'),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

        # 2D mid-slices
        if Ensemble_Toggle is True:
            ix, iy, iz = nx // 2, ny // 2, nz // 2

            x_min, x_max = numpy.min(X), numpy.max(X)
            y_min, y_max = numpy.min(Y), numpy.max(Y)
            z_min, z_max = numpy.min(Z), numpy.max(Z)

            slices = [
                ("XY Slice (Z Mid)", true_field[:, :, iz], pred_field[:, :, iz], error_field[:, :, iz],
                 [x_min, x_max, y_min, y_max], "X", "Y"),
                ("XZ Slice (Y Mid)", true_field[:, iy, :], pred_field[:, iy, :], error_field[:, iy, :],
                 [x_min, x_max, z_min, z_max], "X", "Z"),
                ("YZ Slice (X Mid)", true_field[ix, :, :], pred_field[ix, :, :], error_field[ix, :, :],
                 [y_min, y_max, z_min, z_max], "Y", "Z"),
            ]

            Cmap_Viridis = plt.get_cmap('viridis').copy()
            Cmap_Viridis.set_bad(color='white')
            Cmap_Hot = plt.get_cmap('hot').copy()
            Cmap_Hot.set_bad(color='white')

            for title2, true_slice, pred_slice, err_slice, extent, xlabel, ylabel in slices:
                fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

                true_masked = numpy.ma.masked_where(numpy.abs(true_slice) <= Zero_Tol, true_slice)
                pred_masked = numpy.ma.masked_where(numpy.abs(pred_slice) <= Zero_Tol, pred_slice)
                err_masked  = numpy.ma.masked_where(err_slice <= Zero_Tol, err_slice)

                im0 = ax2[0].imshow(true_masked.T, origin="lower", aspect="auto", cmap=Cmap_Viridis,
                                    vmin=value_min, vmax=value_max, extent=extent)
                ax2[0].set_title("Ground Truth")
                fig2.colorbar(im0, ax=ax2[0], shrink=0.8, label=key)

                im1 = ax2[1].imshow(pred_masked.T, origin="lower", aspect="auto", cmap=Cmap_Viridis,
                                    vmin=value_min, vmax=value_max, extent=extent)
                ax2[1].set_title("Prediction")
                fig2.colorbar(im1, ax=ax2[1], shrink=0.8, label=key)

                im2 = ax2[2].imshow(err_masked.T, origin="lower", aspect="auto", cmap=Cmap_Hot,
                                    vmin=0, vmax=Error_Max, extent=extent)
                ax2[2].set_title("Absolute Error")
                fig2.colorbar(im2, ax=ax2[2], shrink=0.8, label=f"{key} Error")

                for a in ax2:
                    a.set_xlabel(xlabel)
                    a.set_ylabel(ylabel)

                fig2.suptitle(f"{dataset} (T={Time_Index + 1}) {key} - {title2} (zero-valued cells hidden)")
                plt.savefig(os.path.join(key_dir, f"{dataset}_T{Time_Index + 1}_{key}_{title2.replace(' ', '_')}.png"),
                            dpi=200, bbox_inches="tight")
                plt.close(fig2)

    return


def _Weighted_Confusion(Scores, Labels, Weights, Threshold):
    """Weighted TP / FP / FN / TN at a single threshold."""
    Pred_Pos = Scores >= Threshold
    Is_Pos   = Labels == 1

    TP = float(numpy.sum(Weights[Pred_Pos & Is_Pos]))
    FP = float(numpy.sum(Weights[Pred_Pos & ~Is_Pos]))
    FN = float(numpy.sum(Weights[~Pred_Pos & Is_Pos]))
    TN = float(numpy.sum(Weights[~Pred_Pos & ~Is_Pos]))

    return TP, FP, FN, TN


def _Plot_Confusion(TP, FP, FN, TN, Threshold, Title, Result_Path, File_Name):
    """2x2 confusion matrix, shaded by row share, annotated with weighted counts."""
    Matrix = numpy.array([[TN, FP],
                          [FN, TP]], dtype=numpy.float64)

    Row_Sums = Matrix.sum(axis=1, keepdims=True)
    Row_Sums[Row_Sums == 0.0] = 1.0             
    Normalised = Matrix / Row_Sums

    Figure, Axis = plt.subplots(figsize=(6.0, 5.2))
    Image = Axis.imshow(Normalised, cmap='Blues', vmin=0.0, vmax=1.0)

    Axis.set_xticks([0, 1]); Axis.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    Axis.set_yticks([0, 1]); Axis.set_yticklabels(['Actual Negative', 'Actual Positive'])
    Axis.set_title(f'{Title}\n(Threshold = {Threshold:.3f})', fontsize=13, fontweight='bold')

    for Row in range(2):
        for Column in range(2):
            Text_Colour = 'white' if Normalised[Row, Column] > 0.5 else 'black'
            Axis.text(Column, Row,
                      f'{Matrix[Row, Column]:.4g}\n({Normalised[Row, Column] * 100:.2f}% of row)',
                      ha='center', va='center', color=Text_Colour, fontsize=11)

    Colour_Bar = Figure.colorbar(Image, ax=Axis, shrink=0.85)
    Colour_Bar.set_label('Fraction of Actual Class')

    Figure.tight_layout()
    Figure.savefig(os.path.join(Result_Path, File_Name), dpi=300, bbox_inches='tight')
    plt.close(Figure)

    return


def PlotStatisticsCurve(Predictions, Ground_Truth, Result_Path, Sample_Weight=None,
                        Num_Calibration_Bins: int = 20, Quantile_Bins: bool = False):
    """Full diagnostic set for the pooled voxel predictions:

        ROC_Curve.png              discrimination
        Precision_Recall_Curve.png discrimination under imbalance
        Reliability_Diagram.png    calibration, ECE / MCE
        Score_Distribution.png     score histogram split by class
        Threshold_Sweep.png        precision / recall / F1 / IoU vs threshold
        Confusion_Matrix_0p5.png   counts at the default cut
        Confusion_Matrix_BestF1.png counts at the best-F1 cut
    """

    os.makedirs(Result_Path, exist_ok=True)

    Predictions  = numpy.asarray(Predictions, dtype=numpy.float64).ravel()
    Ground_Truth = numpy.asarray(Ground_Truth, dtype=numpy.float64).ravel()

    gt_binary = (Ground_Truth > 0.5).astype(int)

    if Sample_Weight is None:
        Weights = numpy.ones_like(Predictions, dtype=numpy.float64)
    else:
        Weights = numpy.asarray(Sample_Weight, dtype=numpy.float64).ravel()

    Total_Weight = float(numpy.sum(Weights))

    # ROC
    fpr, tpr, _ = roc_curve(gt_binary, Predictions, sample_weight=Sample_Weight)
    roc_auc = float(roc_auc_score(gt_binary, Predictions, sample_weight=Sample_Weight))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=3, color='darkblue', label=f'Ensemble (ROC AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(Result_Path, "ROC_Curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Precision-Recall
    precision, recall, PR_Thresholds = precision_recall_curve(
        gt_binary, Predictions, sample_weight=Sample_Weight)
    ap        = float(average_precision_score(gt_binary, Predictions, sample_weight=Sample_Weight))
    trapz_auc = float(auc(recall, precision))     # legacy value, kept for comparison

    baseline = float(numpy.sum(Weights * gt_binary) / Total_Weight)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=3, color='darkblue',
            label=f'Ensemble (PR AUC = {ap:.3f})')
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5,
               label=f'Baseline (Prevalence = {baseline:.3g})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(Result_Path, "Precision_Recall_Curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Reliability diagram
    if Quantile_Bins:
        Edges = numpy.unique(numpy.quantile(
            Predictions, numpy.linspace(0.0, 1.0, Num_Calibration_Bins + 1)))
        if Edges.size < 2:
            Edges = numpy.array([0.0, 1.0])     # degenerate: all scores identical
    else:
        Edges = numpy.linspace(0.0, 1.0, Num_Calibration_Bins + 1)

    Num_Bins  = Edges.size - 1
    Bin_Index = numpy.clip(numpy.digitize(Predictions, Edges[1:-1], right=False), 0, Num_Bins - 1)

    # Weighted Per-Bin Sums
    Weight_Sum = numpy.bincount(Bin_Index, weights=Weights,               minlength=Num_Bins)
    Label_Sum  = numpy.bincount(Bin_Index, weights=Weights * gt_binary,   minlength=Num_Bins)
    Score_Sum  = numpy.bincount(Bin_Index, weights=Weights * Predictions, minlength=Num_Bins)

    Nonempty   = Weight_Sum > 0
    Mean_Score = numpy.full(Num_Bins, numpy.nan)
    Empirical  = numpy.full(Num_Bins, numpy.nan)
    Mean_Score[Nonempty] = Score_Sum[Nonempty] / Weight_Sum[Nonempty]
    Empirical[Nonempty]  = Label_Sum[Nonempty] / Weight_Sum[Nonempty]

    # Gap Measured against the Bin's Mean Score
    Gap = numpy.abs(Mean_Score[Nonempty] - Empirical[Nonempty])
    ece = float(numpy.sum(Weight_Sum[Nonempty] * Gap) / Total_Weight)
    mce = float(numpy.max(Gap)) if Gap.size else float('nan')
    brier = float(brier_score_loss(gt_binary, numpy.clip(Predictions, 0.0, 1.0),
                                   sample_weight=Sample_Weight))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Mean_Score[Nonempty], Empirical[Nonempty], 'o-', linewidth=3, markersize=7,
            color='darkblue', label=f'Ensemble (ECE = {ece:.3f}, MCE = {mce:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Empirical Frequency (Actual Positive Rate)', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0])

    # Occupancy Bars
    if not Quantile_Bins:
        Axis_Counts = ax.twinx()
        Bin_Centres = 0.5 * (Edges[:-1] + Edges[1:])
        Bin_Widths  = numpy.diff(Edges)
        Axis_Counts.bar(Bin_Centres, Weight_Sum / Total_Weight,
                        width=numpy.maximum(Bin_Widths * 0.9, 1e-4),
                        color='grey', alpha=0.25, zorder=0)
        Axis_Counts.set_ylabel('Fraction of Voxels per Bin', fontsize=10)
        Axis_Counts.set_ylim(0, max(float(numpy.max(Weight_Sum / Total_Weight)) * 4.0, 1e-6))
        Axis_Counts.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(Result_Path, "Reliability_Diagram.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Score Distribution by Class
    Hist_Edges  = numpy.linspace(0.0, 1.0, 51)
    Is_Positive = gt_binary == 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(Predictions[~Is_Positive], bins=Hist_Edges, weights=Weights[~Is_Positive],
            color='steelblue', alpha=0.6, label='Ground Truth Negative')
    ax.hist(Predictions[Is_Positive], bins=Hist_Edges, weights=Weights[Is_Positive],
            color='darkorange', alpha=0.6, label='Ground Truth Positive')
    ax.axvline(0.5, color='k', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    ax.set_yscale('log')                        # class counts differ by orders of magnitude
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Weighted Voxel Count (log scale)', fontsize=12)
    ax.set_title('Predicted Score Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(Result_Path, "Score_Distribution.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Metrics vs Threshold
    Precision_T = precision[:-1]
    Recall_T    = recall[:-1]

    Denominator = Precision_T + Recall_T
    F1_T  = numpy.zeros_like(Denominator)
    Valid = Denominator > 0
    F1_T[Valid] = 2.0 * Precision_T[Valid] * Recall_T[Valid] / Denominator[Valid]

    # IoU
    IoU_T     = numpy.zeros_like(F1_T)
    IoU_Valid = (2.0 - F1_T) > 0
    IoU_T[IoU_Valid] = F1_T[IoU_Valid] / (2.0 - F1_T[IoU_Valid])

    if F1_T.size:
        Best_Index     = int(numpy.argmax(F1_T))
        Best_Threshold = float(PR_Thresholds[Best_Index])
        Best_F1        = float(F1_T[Best_Index])
        Best_IoU       = float(IoU_T[Best_Index])
    else:
        Best_Threshold, Best_F1, Best_IoU = 0.5, float('nan'), float('nan')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(PR_Thresholds, Precision_T, linewidth=2,   color='tab:blue',   label='Precision')
    ax.plot(PR_Thresholds, Recall_T,    linewidth=2,   color='tab:green',  label='Recall')
    ax.plot(PR_Thresholds, F1_T,        linewidth=2.5, color='tab:red',    label='F-Score')
    ax.plot(PR_Thresholds, IoU_T,       linewidth=2,   color='tab:purple', label='IoU')
    ax.axvline(0.5, color='k', linestyle=':', linewidth=1.2, label='Default (0.5)')
    ax.axvline(Best_Threshold, color='k', linestyle='--', linewidth=1.5,
               label=f'Best F1 @ {Best_Threshold:.3f} (F1 = {Best_F1:.3f})')
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Metrics vs. Decision Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='lower center', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(Result_Path, "Threshold_Sweep.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Confusion Matrices at the Default and Best-F1 Threshold
    TP_Default, FP_Default, FN_Default, TN_Default = _Weighted_Confusion(
        Predictions, gt_binary, Weights, 0.5)
    _Plot_Confusion(TP_Default, FP_Default, FN_Default, TN_Default, 0.5,
                    'Confusion Matrix', Result_Path, 'Confusion_Matrix_0p5.png')

    TP_Best, FP_Best, FN_Best, TN_Best = _Weighted_Confusion(
        Predictions, gt_binary, Weights, Best_Threshold)
    _Plot_Confusion(TP_Best, FP_Best, FN_Best, TN_Best, Best_Threshold,
                    'Confusion Matrix (Best F1)', Result_Path, 'Confusion_Matrix_BestF1.png')

    IoU_Denominator = TP_Default + FP_Default + FN_Default
    IoU_Default = float(TP_Default / IoU_Denominator) if IoU_Denominator > 0 else float('nan')

    F1_Denominator = 2.0 * TP_Default + FP_Default + FN_Default
    F1_Default = float(2.0 * TP_Default / F1_Denominator) if F1_Denominator > 0 else float('nan')

    Accuracy_Default = float((TP_Default + TN_Default) / Total_Weight)

    print(f"ROC AUC:                {roc_auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"PR trapezoidal AUC:     {trapz_auc:.4f}   (legacy)")
    print(f"Prevalence:             {baseline:.6g}")
    print(f"Brier score:            {brier:.6g}")
    print(f"ECE / MCE:              {ece:.4f} / {mce:.4f}")
    print(f"@0.50  F1 / IoU / Acc:  {F1_Default:.4f} / {IoU_Default:.4f} / {Accuracy_Default:.4f}")
    print(f"Best F1 threshold:      {Best_Threshold:.4f}  (F1 = {Best_F1:.4f}, IoU = {Best_IoU:.4f})")

    Data_Structure = {"fpr": fpr,
                      "tpr": tpr,
                      "AUC": roc_auc,
                      "precision": precision,
                      "recall": recall,
                      "PR_Thresholds": PR_Thresholds,
                      "PR_AUC": ap,
                      "PR_AUC_Trapezoidal": trapz_auc,
                      "Prevalence": baseline,
                      "Bin_Edges": Edges,
                      "Bin_Mean_Score": Mean_Score,
                      "Empirical_Frequency": Empirical,
                      "Bin_Weight": Weight_Sum,
                      "ECE": ece,
                      "MCE": mce,
                      "Brier": brier,
                      "Threshold_F1": F1_T,
                      "Threshold_IoU": IoU_T,
                      "Best_Threshold": Best_Threshold,
                      "Best_F1": Best_F1,
                      "Best_IoU": Best_IoU,
                      "Confusion_Default": numpy.array([[TN_Default, FP_Default],
                                                        [FN_Default, TP_Default]]),
                      "Confusion_Best": numpy.array([[TN_Best, FP_Best],
                                                     [FN_Best, TP_Best]]),
                      "F1_Default": F1_Default,
                      "IoU_Default": IoU_Default,
                      "Accuracy_Default": Accuracy_Default}
    savemat(os.path.join(Result_Path, 'Performance_Data.mat'), Data_Structure)

    return Data_Structure