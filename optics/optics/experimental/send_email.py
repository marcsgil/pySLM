"""
A function by Laurynas to send e-mails from a python script.

Note that this relies on Laurynas' account, which seems to require two-factor authentication now.
todo: Are there any UoD servers we can use without this issue?
"""
import smtplib
import ssl


def send_email(message, receiver_email: str = 'auto.communication.protocol@gmail.com', wrap_message: bool = True):
    port = 465
    context = ssl.create_default_context()
    output_email = 'auto.communication.protocol@gmail.com'
    password = 'WorldLeading'

    if wrap_message:
        message = f'Subject: Computer-Mail \n\nMessage to the lab user.\n\n\n{message}\n\n\nBest Wishes,\n\nLab Computer'

    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(output_email, password)
        server.sendmail(output_email, receiver_email, message)


if __name__ == '__main__':
    from optics.experimental import log

    send_email('Test')
