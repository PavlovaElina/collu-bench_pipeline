#lang racket
(define (minSubArraySum nums)
  (let loop ([ns (cdr nums)] [cur (car nums)] [best (car nums)])
    (cond [(null? ns) best]
          [else (define c (min (car ns) (+ cur (car ns))))
                (loop (cdr ns) c (min best c))])))
