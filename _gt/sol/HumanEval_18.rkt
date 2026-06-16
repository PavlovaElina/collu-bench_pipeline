#lang racket
(define (how_many_times str sub)
  (define n (string-length str))
  (define m (string-length sub))
  (cond [(= m 0) 0]
        [else (for/sum ([i (in-range (+ (- n m) 1))])
                (if (string=? (substring str i (+ i m)) sub) 1 0))]))
