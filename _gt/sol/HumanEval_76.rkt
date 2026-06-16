#lang racket
(define (is_simple_power x n)
  (cond [(= x 1) #t]
        [(<= n 1) #f]
        [else (let loop ([p n])
                (cond [(= p x) #t]
                      [(> p x) #f]
                      [else (loop (* p n))]))]))
