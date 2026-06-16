#lang racket
(define (below_zero ops)
  (let loop ([ops ops] [bal 0])
    (cond [(null? ops) #f]
          [else (define nb (+ bal (car ops)))
                (if (< nb 0) #t (loop (cdr ops) nb))])))
