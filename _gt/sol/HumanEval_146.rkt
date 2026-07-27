#lang racket
(define (specialFilter nums)
  (for/sum ([x (in-list nums)]
            #:when (and (> x 10)
                        (let ([s (number->string x)])
                          (and (odd? (- (char->integer (string-ref s 0)) 48))
                               (odd? (- (char->integer (string-ref s (- (string-length s) 1))) 48))))))
    1))
