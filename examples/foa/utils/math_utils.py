def extract_last_boxed_answer(text):
    answers = [""]
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    last_boxed = answers[-1]  # Take the last one
    return last_boxed


def naive_parse(answer: str) -> str:
    out = []
    start = False
    end = False
    for l in list(answer)[::-1]:
        if l in "0123456789" and not end:
            start = True
            if l != ',':
                out.append(l)
        else:
            if start:
                end = True
    out = out[::-1]
    return "".join(out)


def extract_answer_from_output(raw_output):
    output = extract_last_boxed_answer(raw_output)
    if output == "":
        output = naive_parse(raw_output)
    return output
