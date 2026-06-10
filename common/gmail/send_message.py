from .gmail_sender import GmailSender

def send_message(
        subject,
        body
):
    sender = GmailSender()
    sender_email = 'kikuchi.riku.s2@dc.tohoku.ac.jp'
    receiver_email = 'kikuchi.riku.s2@dc.tohoku.ac.jp'

    if sender.service:
        sender.send_message(
            sender_email,
            receiver_email,
            subject,
            body
        )
    return


def send_progress_message(
        subject,
        filename,
        dict_progress,
        comment=None
):
    body = f'{filename}\n' + f'Progress: {dict_progress["percent"]:.2f}%, Time/Iter: {dict_progress["time_per_iter"]:.2f} s, ETA: {dict_progress["eta"]}\n'
    body = body + '-' * 20 + '\n'
    if comment is not None:
        body = body + comment
    send_message(subject, body)
    return
