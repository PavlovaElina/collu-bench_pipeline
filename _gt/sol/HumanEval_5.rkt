#lang racket
(define (intersperse numbers delimeter)
  (cond [(null? numbers) (list)]
        [else (cons (car numbers)
                    (append-map (lambda (x) (list delimeter x)) (cdr numbers)))]))
