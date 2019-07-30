

def writer_training(writer, loss,  epoch):
    writer.add_scalar("Train/Loss", loss, epoch)
    #To Be Added

    return writer


def writer_validation(writer, loss, epoch):
    writer.add_scalar("Val/Loss", loss, epoch)
    # To Be Added

    return writer


def writer_arguments(writer, args, epoch):
    attrs = vars(args)
    for key, val in attrs.items():
        writer.add_text('Parameter', "{}: {}".format(key, val), epoch)
