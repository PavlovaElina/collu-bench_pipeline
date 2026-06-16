#lang racket
(define (correct_bracketing brackets)
  (let loop ([chars (string->list brackets)] [depth 0])
    (cond [(< depth 0) #f]
          [(null? chars) (= depth 0)]
          [(char=? (car chars) #\<) (loop (cdr chars) (+ depth 1))]
          [else (loop (cdr chars) (- depth 1))])))
