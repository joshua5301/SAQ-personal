# import core.asam as asam
import utils.asam as asam
import utils.qasam as qasam
# import utils.qasam as qasam
import utils.qsam as qsam
import utils.sam as sam
import utils.flipsam as flipsam
import utils.tilted_sr as tilted_sr
import utils.flipqsam as flipqsam
import utils.kltilt as kltilt
import utils.logitflip as logitflip
import utils.gridusam as gridusam


def get_minimizer(model, optimizer, args):
    if "GridUSAM" in args.opt_type:
        minimizer = gridusam.GridUSAM(
            optimizer,
            model,
            rho=args.rho,
            space=args.space,
            perturb_continuous=args.perturb_continuous,
        )
    elif "LogitFlip" in args.opt_type:
        minimizer = logitflip.LogitFlip(
            optimizer,
            model,
            tau=args.tau,
            scope=args.flip_scope,
            perturb_continuous=args.perturb_continuous,
            rho=args.rho,
        )
    elif "KLTilt" in args.opt_type:
        minimizer = kltilt.KLTilt(
            optimizer,
            model,
            tau=args.tau,
            deterministic=args.deterministic,
            perturb_continuous=args.perturb_continuous,
            rho=args.rho,
        )
    elif "TiltedSR" in args.opt_type:
        minimizer = tilted_sr.TiltedSR(
            optimizer,
            model,
            beta=args.beta,
            scale_mode=args.scale_mode,
            deterministic=args.deterministic,
            perturb_continuous=args.perturb_continuous,
            rho=args.rho,
        )
    elif "FlipQSAM" in args.opt_type:
        minimizer = flipqsam.FlipQSAM(
            optimizer,
            model,
            rho_flip=args.rho_flip,
            rho=args.rho,
            perturb_continuous=args.perturb_continuous,
        )
    elif "FlipSAM" in args.opt_type:
        minimizer = flipsam.FlipSAM(
            optimizer,
            model,
            kappa=args.kappa,
            kappa_mode=args.kappa_mode,
            perturb_continuous=args.perturb_continuous,
            cont_radius=args.cont_radius,
            rho=args.rho,
        )
    elif "QSAM" in args.opt_type:
        minimizer = qsam.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "QASAM" in args.opt_type:
        minimizer = qasam.QASAM(
            optimizer,
            model,
            rho=args.rho,
            eta=args.eta,
            include_wclip=args.include_wclip,
            include_aclip=args.include_aclip,
            include_bn=args.include_bn,
        )
    elif "ASAM" in args.opt_type:
        minimizer = asam.ASAM(optimizer, model, rho=args.rho, eta=args.eta)
    elif "SAM" in args.opt_type:
        SAM = sam.SAM
        minimizer = SAM(optimizer, model, rho=args.rho, eta=args.eta,)
    else:
        raise NotImplemented

    return minimizer
