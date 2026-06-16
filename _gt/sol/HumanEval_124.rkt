#lang racket
(define (valid_date date)
  (define parts (string-split date "-"))
  (cond [(not (= (length parts) 3)) #f]
        [(not (andmap (lambda (p) (and (> (string-length p) 0)
                                       (andmap char-numeric? (string->list p)))) parts)) #f]
        [else
         (define mm (string->number (first parts)))
         (define dd (string->number (second parts)))
         (cond [(or (< mm 1) (> mm 12)) #f]
               [(member mm '(1 3 5 7 8 10 12)) (and (>= dd 1) (<= dd 31))]
               [(member mm '(4 6 9 11)) (and (>= dd 1) (<= dd 30))]
               [else (and (>= dd 1) (<= dd 29))])]))
